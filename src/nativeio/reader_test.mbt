///|
test "native reader rejects embedded NUL and missing files" {
  assert_true(
    @nativeio.inspect_file("x\u0000y", @gguf.default_limits()) is Err(_),
  )
  assert_true(
    @nativeio.inspect_file(
      "/__llmcalc_missing_test_directory__/missing.gguf",
      @gguf.default_limits(),
    )
    is Err(_),
  )
}
