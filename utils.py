from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

DATASET_PATH = "data/datasets/"
FSELECTION_PATH = "data/fSelection/"
MODELS_COMPARISON_PATH = "data/modelsComparison/"
MLP_MAX_ITER = 500
PR_MAX_ITER = 300
LR_MAX_ITER = 5000  #idk
SVC_MAX_ITER = 1000 ## what do be good values...
KNN_K = 5
RANDOM_STATE = 42
TEST_SIZE = 0.3

# String features
string_features = [
    "proto",
    "service",
    "conn_state",
    "src_ip",
    "dst_ip",
    "dns_query",
    "ssl_version",
    "ssl_cipher",
    "ssl_subject",
    "ssl_issuer",
    "http_method",
    "http_uri",
    "http_version",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "weird_name",
    "weird_addl",
    "http_user_agent",
    "type",
]
# Number features
int_features = [
    "src_bytes",
    "dst_bytes",
    "dst_port",
    "src_port",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "dns_qclass",
    "dns_qtype",
    "dns_rcode",
    "http_trans_depth",
    "http_request_body_len",
    "http_response_body_len",
    "http_status_code",
]
float_features = ["duration"]
# Boolean features
boolean_features = [
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_resumed",
    "ssl_established",
    "weird_notice",
    "label",
]

toniot_dtype_spec = {}
for feature in string_features:
    toniot_dtype_spec[feature] = "string"
for feature in int_features:
    toniot_dtype_spec[feature] = "Int64"
for feature in float_features:
    toniot_dtype_spec[feature] = "float64"
for feature in boolean_features:
    toniot_dtype_spec[feature] = "boolean"

models = [
    (
        "RandomForest",
        RandomForestClassifier(n_jobs=-1),
    ),  # https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    (
        "DecisionTree",
        DecisionTreeClassifier(),
    ),  # https://scikit-learn.org/1.5/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    (
        "KNN",
        KNeighborsClassifier(),
    ),  # https://scikit-learn.org/1.5/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    (
        "MLPClassifier",
        MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=MLP_MAX_ITER),
    ),  # https://scikit-learn.org/1.5/modules/generated/sklearn.neural_network.MLPClassifier.html
    (
        "Perceptron",
        Perceptron(max_iter=PR_MAX_ITER),
    ),  # https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Perceptron.html
    (
        "XGBoost",
        xgb.XGBClassifier(),
    ),  # https://xgboost.readthedocs.io/en/latest/python/python_api.html
]
