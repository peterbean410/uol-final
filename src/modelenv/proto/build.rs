fn main() {
    // Compile the .proto files
    tonic_build::compile_protos("proto/environment.proto")
        .expect("Failed to compile protobuf files");

    // Configure prost to use bytes instead of String for binary data
    println!("cargo:rustc-env=PROST_CONFIG=bytes");
}
