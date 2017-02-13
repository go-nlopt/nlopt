load("@io_bazel_rules_go//go:def.bzl", "cgo_library", "go_library", "go_prefix", "go_test")

go_prefix("github.com/go-nlopt/nlopt")

cgo_library(
    name = "cgo_default_library",
    srcs = [
        "nlopt.go",
        "nlopt.h",
        "nlopt_cfunc.go",
    ],
    clinkopts = ["-lnlopt"],
    copts = [
        "-Os",
        "-fno-common",
        "-mtune=native",
        "-march=native",
    ],
    visibility = ["//visibility:private"],
)

go_library(
    name = "go_default_library",
    srcs = ["cfunc_reg.go"],
    library = ":cgo_default_library",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["nlopt_test.go"],
    library = ":go_default_library",
)
