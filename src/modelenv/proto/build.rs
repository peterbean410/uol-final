fn main() {
    // Compile the .proto files
    tonic_build::compile_protos("proto/environment.proto")
        .expect("Failed to compile protobuf files");

    // cTrader Open API messages (proto2, no gRPC services). Vendored under
    // proto/proto/ctrader/, reconstructed faithfully from the ctrader_open_api
    // descriptors the production Python trader already uses, so the generated
    // Rust types match the live wire format exactly. The empty-package types
    // are included under the `ctrader` module in lib.rs.
    tonic_build::configure()
        .build_server(false)
        .build_client(false)
        .compile_protos(
            &[
                "proto/ctrader/OpenApiCommonMessages.proto",
                "proto/ctrader/OpenApiCommonModelMessages.proto",
                "proto/ctrader/OpenApiModelMessages.proto",
                "proto/ctrader/OpenApiMessages.proto",
            ],
            &["proto/ctrader"],
        )
        .expect("Failed to compile cTrader protobuf files");

    // Configure prost to use bytes instead of String for binary data
    println!("cargo:rustc-env=PROST_CONFIG=bytes");
}
