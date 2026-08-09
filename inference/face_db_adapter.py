"""通过文件名加载 42_face_db.py，GUI 只依赖其高层函数。"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


_DB_PATH = Path(__file__).resolve().parent.parent / "42_face_db.py"
_SPEC = spec_from_file_location("verified_face_db", str(_DB_PATH))
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("无法加载人脸库模块：%s" % _DB_PATH)
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

YUNET = _MODULE.YUNET
SFACE = _MODULE.SFACE
DEVICE_ID = _MODULE.DEVICE_ID
MATCH_CONF_TH = _MODULE.MATCH_CONF_TH
ENROLL_CONF_TH = _MODULE.ENROLL_CONF_TH
extract_feature_for_enroll = _MODULE.extract_feature_for_enroll
extract_feature_for_match = _MODULE.extract_feature_for_match
build_index = _MODULE.build_index
detect_face = _MODULE.detect_face
enroll_person = _MODULE.enroll_person
match_feature = _MODULE.match_feature
list_enrolled = _MODULE.list_enrolled
delete_person = _MODULE.delete_person
