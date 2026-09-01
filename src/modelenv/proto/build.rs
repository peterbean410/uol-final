fn main() {
        tonic_build::compile_protos("proto/environment.proto")
        .expect("Failed to compile protobuf files");

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

        println!("cargo:rustc-env=PROST_CONFIG=bytes");
}
