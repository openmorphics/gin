fn main() {
    #[cfg(target_os = "macos")]
    {
        // Allow unresolved Python symbols to be resolved at runtime by the Python interpreter.
        println!("cargo:rustc-cdylib-link-arg=-undefined");
        println!("cargo:rustc-cdylib-link-arg=dynamic_lookup");
    }
}