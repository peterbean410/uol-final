include!(concat!(env!("OUT_DIR"), "/environment.rs"));

/// cTrader Open API messages (proto2), generated from the vendored protos under
/// `proto/proto/ctrader/`. These carry no `package`, so prost emits them to the
/// empty-package file `_.rs`; expose them under a `ctrader` module so the
/// `ProtoOa*` request/response/event types are reachable as
/// `modelenv_proto::ctrader::ProtoOaNewOrderReq`, etc.
pub mod ctrader {
    include!(concat!(env!("OUT_DIR"), "/_.rs"));
}
