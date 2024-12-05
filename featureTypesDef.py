# String features
string_features = [
    "proto", "service", "conn_state",
    "dns_query", "ssl_version", "ssl_cipher", "ssl_subject", "ssl_issuer",
    "http_method", "http_uri", "http_version", "http_orig_mime_types",
    "http_resp_mime_types", "weird_name", "weird_addl", "http_user_agent"
]
# Number features
int_features = [
    "src_bytes", "dst_bytes",
    "missed_bytes", "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes",
    "dns_qclass", "dns_qtype", "dns_rcode", "http_trans_depth",
    "http_request_body_len", "http_response_body_len", "http_status_code"
]
float_features = ["duration"]
# Boolean features
boolean_features = [
    "dns_AA", "dns_RD", "dns_RA", "dns_rejected", "ssl_resumed",
    "ssl_established", "weird_notice"
]

toniot_dtype_spec = {}
for feature in string_features:
        toniot_dtype_spec[feature] = 'string'
for feature in int_features:
        toniot_dtype_spec[feature] = 'Int64'
for feature in float_features:
        toniot_dtype_spec[feature] = 'float64'
for feature in boolean_features:
        toniot_dtype_spec[feature] = 'boolean'