"""KIE (Processing Layer) — 2-tier, multi-model.

Tier-1 (this file): candidate generation (regex/keyword/layout-graph) ->
feature vector -> scikit-learn field classifier -> calibrated confidence.
This is a LEARNED model, not pure regex (addresses the production requirement).
Tier-2 (router.py + vlm): VLM OCR-free fallback for hard cases.
"""  # Mô tả vai trò của file KIE trong pipeline.
from __future__ import annotations  # Bật forward references cho type hints.
import re  # Dùng regex để nhận diện ngày / tiền / mã.
import joblib  # Lưu và nạp model đã train.
from .config import ALL_FIELDS  # Danh sách tất cả field cần trích xuất.

KIE_VERSION_FALLBACK = "rule-only-baseline"  # Version mặc định khi chưa có model học.

# ---- anchors (VN + EN) ----------------------------------------------------
ANCHORS = {  # Từ khóa gợi ý theo từng field.
    "merchant_name": [],  # Merchant name thường không có anchor cố định.
    "date": ["date", "ngay", "ngày"],  # Anchor cho trường ngày.
    "total_amount": ["total", "amount", "sum", "tong cong", "tổng cộng",  # Anchor cho tổng tiền.
                     "thanh tien", "thành tiền", "tong", "tổng"],  # Biến thể tiếng Việt / tiếng Anh.
    "invoice_id": ["invoice", "bill", "receipt", "no", "hoa don", "hóa đơn",  # Anchor cho mã hóa đơn.
                   "so hd", "số hđ", "ma gd", "mã gd"],  # Các cách viết khác nhau.
    "payment_method": ["cash", "card", "qr", "visa", "momo", "tien mat",  # Anchor cho phương thức thanh toán.
                       "tiền mặt", "the", "thẻ", "payment", "chuyen khoan"],  # Các biến thể phổ biến.
}  # Kết thúc bảng anchor.
DATE_RE = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")  # Regex nhận diện ngày.
ID_RE = re.compile(r"[A-Z]{1,4}\d{3,}")  # Regex nhận diện mã dạng chữ + số.
# Money: thousands-grouped (VND "235,000"), decimal cents (SROIE "9.00",
# "1,234.56"), or a bare >=3-digit run. Dates are stripped first so "25.12.2018"
# is not parsed as money. Convention: ',' = thousands (drop), '.' = decimal.
MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d{1,2})?|\d+\.\d{2}|\d{3,}")  # Regex nhận diện tiền.


# ---- shared normalization (MUST match between train / infer / eval) -------
def norm_money(s: str):  # Chuẩn hóa chuỗi tiền sang float.
    """Return a canonical float amount, or None. Robust across VND/decimal."""
    t = DATE_RE.sub(" ", s or "")  # Xóa phần ngày để không nhầm ngày thành tiền.
    m = MONEY_RE.search(t)  # Tìm mẫu tiền trong chuỗi.
    if not m:  # Không có gì khớp thì trả về None.
        return None  # Không tìm thấy tiền.
    raw = m.group().replace(",", "")  # Bỏ dấu phẩy vì là phân tách hàng nghìn.
    try:  # Thử chuyển sang số thực.
        return round(float(raw), 2)  # Làm tròn 2 chữ số thập phân.
    except ValueError:  # Nếu parse thất bại thì trả về None.
        return None  # Giá trị không hợp lệ.


def norm_date(s: str):  # Chuẩn hóa ngày sang YYYY-MM-DD.
    m = DATE_RE.search(s or "")  # Tìm chuỗi ngày.
    if not m:  # Không có ngày thì trả về None.
        return None  # Không tìm thấy ngày.
    parts = re.split(r"[/\-.]", re.sub(r"\s", "", m.group()))  # Tách thành d/m/y.
    if len(parts) != 3:  # Nếu không đủ 3 phần thì bỏ.
        return None  # Định dạng ngày không hợp lệ.
    d, mth, y = parts  # Gán ngày/tháng/năm.
    y = ("20" + y) if len(y) == 2 else y  # Nếu năm 2 chữ số thì giả định thế kỷ 21.
    try:  # Thử format lại theo chuẩn.
        return f"{int(y):04d}-{int(mth):02d}-{int(d):02d}"  # Trả về YYYY-MM-DD.
    except ValueError:  # Nếu số không hợp lệ thì bỏ.
        return None  # Không chuẩn hóa được.


def norm_text(s):  # Chuẩn hóa text thường: trim, gộp khoảng trắng, lowercase.
    return re.sub(r"\s+", " ", (s or "").strip()).lower()  # Chuỗi đã chuẩn hóa.


def norm_field(field: str, value: str):  # Chuẩn hóa theo từng field.
    if field == "total_amount":  # Nếu field là tổng tiền.
        return norm_money(value)  # Trả về số tiền đã chuẩn hóa.
    if field == "date":  # Nếu field là ngày.
        return norm_date(value)  # Trả về ngày chuẩn hóa.
    if field == "payment_method":  # Nếu field là phương thức thanh toán.
        t = norm_text(value)  # Chuẩn hóa text trước khi dò keyword.
        for kw in ANCHORS["payment_method"]:  # Duyệt từng anchor của payment_method.
            if kw in t:  # Nếu keyword xuất hiện trong text.
                return {"qr": "QR", "card": "CARD", "visa": "CARD", "cash": "CASH",  # Map keyword sang nhãn chuẩn.
                        "tien mat": "CASH", "tiền mặt": "CASH", "momo": "MOMO"}.get(kw, kw.upper())  # Fallback lên uppercase.
        return None  # Không khớp keyword nào.
    if field == "invoice_id":  # Nếu field là mã hóa đơn.
        m = ID_RE.search((value or "").upper())  # Tìm mẫu mã trên chuỗi viết hoa.
        return m.group() if m else None  # Trả về mã nếu có.
    return norm_text(value)  # merchant_name: chỉ cần normalize text thường.


# ---- feature extraction ---------------------------------------------------
def _kw_proximity(token, field, tokens):  # Tính độ gần token đến anchor keyword của field.
    """1 - normalized distance from token to nearest field anchor keyword."""
    anchors = ANCHORS[field]  # Lấy danh sách anchor của field.
    if not anchors:  # Nếu không có anchor thì không chấm điểm proximity.
        return 0.0  # Không có ngữ cảnh anchor.
    tx = (token["bbox"][0] + token["bbox"][2]) / 2  # Tọa độ x trung tâm token.
    ty = (token["bbox"][1] + token["bbox"][3]) / 2  # Tọa độ y trung tâm token.
    best = 1e9  # Khởi tạo khoảng cách lớn nhất có thể.
    for t in tokens:  # Duyệt mọi token để tìm anchor gần nhất.
        low = t["text"].lower()  # Lấy text lowercase.
        if any(a in low for a in anchors):  # Nếu token này chứa anchor.
            ax = (t["bbox"][0] + t["bbox"][2]) / 2  # X của anchor token.
            ay = (t["bbox"][1] + t["bbox"][3]) / 2  # Y của anchor token.
            best = min(best, abs(tx - ax) + abs(ty - ay))  # Lấy khoảng cách Manhattan nhỏ nhất.
    if best > 1e8:  # Nếu không có anchor nào tìm thấy.
        return 0.0  # Score proximity bằng 0.
    return max(0.0, 1.0 - best / 1000.0)  # Chuẩn hóa thành điểm trong [0, 1].


def group_lines(tokens):  # Gom các token cùng hàng thành một candidate dòng.
    """Layout-graph: merge tokens sharing a text row into one line candidate.

    Fixes the train/serve gap where OCR splits a multi-word title ("ABC MART")
    into separate tokens. Training uses line-level tokens, so grouping makes
    inference candidates consistent with training.
    """
    if not tokens:  # Không có token thì trả về list rỗng.
        return []  # Không có candidate.
    toks = sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2)  # Sắp theo trục y.
    lines, cur = [], [toks[0]]  # Danh sách dòng và dòng hiện tại.
    for t in toks[1:]:  # Duyệt token còn lại.
        cy = (t["bbox"][1] + t["bbox"][3]) / 2  # Tâm y của token hiện tại.
        ref = cur[-1]  # Token tham chiếu cuối cùng trong dòng.
        ref_cy = (ref["bbox"][1] + ref["bbox"][3]) / 2  # Tâm y của token tham chiếu.
        ref_h = ref["bbox"][3] - ref["bbox"][1]  # Chiều cao token tham chiếu.
        if abs(cy - ref_cy) <= 0.6 * max(ref_h, 1):  # Nếu cùng hàng tương đối.
            cur.append(t)  # Gộp vào dòng hiện tại.
        else:  # Nếu sang dòng mới.
            lines.append(cur)  # Lưu dòng cũ.
            cur = [t]  # Bắt đầu dòng mới.
    lines.append(cur)  # Lưu dòng cuối.
    merged = []  # Kết quả token đã ghép.
    for grp in lines:  # Duyệt từng nhóm token cùng dòng.
        grp = sorted(grp, key=lambda t: t["bbox"][0])  # Sắp theo x từ trái sang phải.
        merged.append({  # Tạo token dòng mới.
            "text": " ".join(g["text"].strip() for g in grp),  # Ghép text bằng dấu cách.
            "bbox": [min(g["bbox"][0] for g in grp), min(g["bbox"][1] for g in grp),  # Bbox bao toàn dòng.
                     max(g["bbox"][2] for g in grp), max(g["bbox"][3] for g in grp)],  # Cạnh phải / dưới.
            "conf": min(g["conf"] for g in grp),  # Confidence lấy nhỏ nhất trong dòng.
        })  # Kết thúc một candidate dòng.
    return merged  # Trả về list token đã gộp.


def token_features(token, field_idx, tokens, W, H, max_money, max_height):  # Tạo vector đặc trưng cho token.
    txt = token["text"]  # Text thô của token.
    cx = (token["bbox"][0] + token["bbox"][2]) / 2  # Tâm x.
    cy = (token["bbox"][1] + token["bbox"][3]) / 2  # Tâm y.
    height = token["bbox"][3] - token["bbox"][1]  # Chiều cao bbox.
    digits = sum(c.isdigit() for c in txt)  # Đếm chữ số trong text.
    money_val = norm_money(txt) if MONEY_RE.search(txt) else None  # Giá trị tiền nếu parse được.
    feats = [  # Các đặc trưng số cơ bản.
        token["conf"],  # Confidence OCR.
        cy / max(H, 1),                       # Vị trí dọc (merchant thường ở trên, total ở dưới).
        cx / max(W, 1),                       # Vị trí ngang.
        height / max(H, 1),                   # Kích thước chữ tương đối.
        digits / max(len(txt), 1),            # Tỷ lệ chữ số.
        1.0 if MONEY_RE.search(txt) else 0.0,  # Có mẫu tiền hay không.
        1.0 if DATE_RE.search(txt) else 0.0,  # Có mẫu ngày hay không.
        min(len(txt), 40) / 40.0,  # Độ dài text chuẩn hóa.
        1.0 if (money_val is not None and max_money and money_val >= max_money) else 0.0,  # Có phải tiền lớn nhất không.
        _kw_proximity(token, ALL_FIELDS[field_idx], tokens),  # Độ gần anchor của field.
        1.0 if (max_height and height >= max_height - 1e-6) else 0.0,  # Có phải font lớn nhất không.
    ]  # Kết thúc feature số.
    onehot = [0.0] * len(ALL_FIELDS)  # Vector one-hot theo field.
    onehot[field_idx] = 1.0  # Đánh dấu field hiện tại.
    return feats + onehot  # Nối feature số + one-hot.


def candidates(tokens):  # Tạo candidate sau khi gộp dòng và tính thống kê toàn trang.
    """Layout-graph candidate generation: group into lines, then each line is a candidate."""
    tokens = group_lines(tokens)  # Gộp token theo dòng.
    money_vals = [norm_money(t["text"]) for t in tokens if MONEY_RE.search(t["text"])]  # Các giá trị tiền tìm được.
    max_money = max([m for m in money_vals if m], default=0)  # Tiền lớn nhất trong trang.
    W = max((t["bbox"][2] for t in tokens), default=1)  # Chiều rộng ảnh/tài liệu.
    H = max((t["bbox"][3] for t in tokens), default=1)  # Chiều cao ảnh/tài liệu.
    max_height = max((t["bbox"][3] - t["bbox"][1] for t in tokens), default=1)  # Chiều cao bbox lớn nhất.
    return tokens, W, H, max_money, max_height  # Trả về dữ liệu đã chuẩn bị.


# ---- model wrapper --------------------------------------------------------
class KIEModel:  # Wrapper cho model KIE đã train hoặc baseline rule-only.
    def __init__(self, clf=None, version=KIE_VERSION_FALLBACK):  # Khởi tạo model.
        self.clf = clf  # Classifier sklearn đã train.
        self.version = version  # Version model.

    @classmethod
    def load(cls, path):  # Nạp model từ file joblib.
        obj = joblib.load(path)  # Đọc object đã lưu.
        return cls(clf=obj["clf"], version=obj["version"])  # Tạo instance mới.

    def save(self, path, version):  # Lưu model ra file.
        self.version = version  # Cập nhật version nội bộ.
        joblib.dump({"clf": self.clf, "version": version}, path)  # Ghi xuống đĩa.

    def _score(self, feats):  # Chấm điểm xác suất cho một feature vector.
        if self.clf is None:  # Nếu chưa có classifier.
            return None  # Trả về None để dùng baseline rule.
        return float(self.clf.predict_proba([feats])[0][1])  # Lấy xác suất lớp dương.

    def extract(self, tokens) -> dict:  # Trích xuất toàn bộ field từ danh sách token.
        """Return {field: (value, confidence, route_hint)}."""
        toks, W, H, max_money, max_height = candidates(tokens)  # Chuẩn bị candidate + thống kê.
        out = {}  # Kết quả cuối cùng theo field.
        for fi, field in enumerate(ALL_FIELDS):  # Duyệt từng field cần dự đoán.
            best, best_p = None, -1.0  # Token tốt nhất và điểm cao nhất.
            for t in toks:  # Duyệt từng candidate token/dòng.
                nv = norm_field(field, t["text"])  # Chuẩn hóa text theo field.
                if nv is None:  # Nếu không khớp pattern của field.
                    continue  # Bỏ qua token này.
                feats = token_features(t, fi, toks, W, H, max_money, max_height)  # Tạo feature vector.
                p = self._score(feats)  # Chấm điểm bằng classifier nếu có.
                # rule-only baseline fallback when no classifier
                if p is None:  # Nếu không có model học.
                    p = _rule_score(t, field, toks, W, H, max_money)  # Dùng heuristic baseline.
                if p > best_p:  # Nếu điểm này tốt hơn best hiện tại.
                    best_p, best = p, (nv, t)  # Cập nhật token tốt nhất.
            if best is None:  # Nếu không tìm được token phù hợp.
                out[field] = (None, 0.0)  # Field này không trích được.
            else:  # Nếu có token tốt nhất.
                nv, tok = best  # Tách value chuẩn hóa và token gốc.
                ocr_c = tok["conf"]  # Confidence OCR của token.
                pattern_ok = 1.0  # Token đã qua norm_field nên coi như pattern hợp lệ.
                ens = 0.6 * best_p + 0.4 * (0.5 * ocr_c + 0.5 * pattern_ok)  # Ensemble score giữa model và OCR.
                out[field] = (nv, round(min(1.0, ens), 3))  # Lưu value + confidence đã chặn trên 1.
        return out  # Trả về dict toàn bộ field.


def _rule_score(token, field, tokens, W, H, max_money):  # Điểm heuristic khi không có model học.
    """Heuristic score used as Setting-A baseline (no learned model)."""
    cy = (token["bbox"][1] + token["bbox"][3]) / 2 / max(H, 1)  # Vị trí y đã chuẩn hóa.
    prox = _kw_proximity(token, field, tokens)  # Độ gần anchor keyword.
    if field == "merchant_name":  # Merchant name thường nằm ở phía trên.
        return 1.0 - cy  # Càng gần đầu trang càng tốt.
    if field == "total_amount":  # Total amount cần ưu tiên anchor và số lớn nhất.
        mv = norm_money(token["text"]) or 0  # Giá trị tiền của token hiện tại.
        return 0.5 * prox + 0.5 * (1.0 if max_money and mv >= max_money else 0.0)  # Kết hợp proximity + max money.
    return 0.4 + 0.6 * prox  # Các field khác chủ yếu dựa vào proximity.