load("@io_bazel_rules_go//go:def.bzl", "go_prefix", "go_library", "go_test", "cgo_library")

go_prefix("github.com/go-nlopt/nlopt")

cgo_library(
    name = "cgo_default_library",
    srcs = [
        "nlopt.go",
        "nlopt.h",
        "nlopt_cfunc.go",
    ],
    clinkopts = [
        "-lnlopt",
        "-lm",
    ],
    copts = ["-Os"],
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
