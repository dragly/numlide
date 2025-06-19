from dataclasses import dataclass
from re import I
import halide as hl


_IMPLS_BIG = {}
_IMPLS_SMALL = {}


@dataclass
class Matmul:
    ty: hl.Type
    a: hl.ImageParam
    b: hl.ImageParam
    output: hl.Func


def _create_matmul_big(ty: hl.Type):
    a = hl.ImageParam(ty, 2, "a_param")
    b = hl.ImageParam(ty, 2, "b_param")

    # if a.height() != b.width():
    #     raise ValueError("Inner size does not match: {a.height()=} != {b.width()=}")

    x = hl.Var("x")
    y = hl.Var("y")

    matmul = hl.Func("matmul_impl")
    inner_size = a.width()
    k = hl.RDom([(0, inner_size)])
    matmul[x, y] = hl.cast(ty, 0)
    matmul[x, y] += a[k, y] * b[x, k]
    output = hl.Func("matmul")
    output[x, y] = matmul[x, y]
    output_width = b.width()
    output_height = a.height()
    xy = hl.Var("xy")
    xi = hl.Var("xi")
    yi = hl.Var("yi")
    xo = hl.Var("xo")
    yo = hl.Var("yo")
    yii = hl.Var("yii")
    size_threshold = 32

    # output_big = output.specialize((output_size_x > size_threshold) & (output_size_y > size_threshold) & (inner_size > size_threshold))

    # schedule copied from
    # https://github.com/halide/Halide/blob/bf65d521d69d75c0ffa9459cdf797886b1bc77e2/test/performance/matrix_multiplication.cpp
    target = hl.get_jit_target_from_environment()
    vec = target.natural_vector_size(ty)
    print(f"{vec=}")
    inner_tile_x = 3 * vec
    print(f"{inner_tile_x=}")
    inner_tile_y = 8
    print(f"{inner_tile_y=}")
    tile_y = hl.max(output_width // 4, 4)
    print(f"{tile_y=}")
    tile_k = hl.max(inner_size // 16, 4)
    print(f"{tile_k=}")
    output.tile(
        x,
        y,
        xi,
        yi,
        inner_tile_x,
        tile_y,
        # tail=hl.TailStrategy.GuardWithIf,
    ).split(
        yi,
        yi,
        yii,
        inner_tile_y,
        # tail=hl.TailStrategy.GuardWithIf,
    ).vectorize(
        xi, vec
    ).unroll(
        xi
    ).unroll(
        yii
    ).fuse(
        x, y, xy
    ).parallel(
        xy
    )
    ko = hl.RVar("ko")
    ki = hl.RVar("ki")
    z = hl.Var("z")
    matmul.update().split(
        k,
        ko,
        ki,
        tile_k,
        # tail=hl.TailStrategy.GuardWithIf,
    )
    intm = matmul.update().rfactor(ko, z)

    intm.compute_at(matmul, y).vectorize(x, vec).unroll(x).unroll(y)

    intm.update(0).reorder(x, y, ki).vectorize(x, vec).unroll(x).unroll(y)

    matmul.compute_at(output, xy).vectorize(x, vec).unroll(x)

    matmul.update().split(
        y,
        y,
        yi,
        inner_tile_y,
        # tail=hl.TailStrategy.GuardWithIf,
    ).reorder(x, yi, y, ko).vectorize(x, vec,).unroll(x).unroll(yi)

    output.bound(x, 0, output_width).bound(y, 0, output_height)
    output.compute_root()

    output.compile_to(
        {
            hl.OutputFileType.stmt_html: "output.html",
        },
        [
            a,
            b,
        ],
        "output",
    )

    output.compile_jit()

    return Matmul(
        ty=ty,
        a=a,
        b=b,
        output=output,
    )


def _create_matmul_small(ty: hl.Type):
    a = hl.ImageParam(ty, 2, "a_param")
    b = hl.ImageParam(ty, 2, "b_param")

    x = hl.Var("x")
    y = hl.Var("y")

    matmul = hl.Func("matmul_impl")
    inner_size = a.width()
    k = hl.RDom([(0, inner_size)])
    matmul[x, y] = hl.cast(ty, 0)
    matmul[x, y] += a[k, y] * b[x, k]
    output = hl.Func("matmul")
    output[x, y] = matmul[x, y]
    xi = hl.Var("xi")
    yi = hl.Var("yi")
    xo = hl.Var("xo")
    yo = hl.Var("yo")
    tile = hl.Var("tile")

    output.tile(
        x,
        y,
        xo,
        yo,
        xi,
        yi,
        32,
        32,
        tail=hl.TailStrategy.GuardWithIf,
    ).vectorize(xi, 4).fuse(
        xo, yo, tile
    ).parallel(tile)
    output.compute_root()

    output.compile_jit()

    return Matmul(
        ty=ty,
        a=a,
        b=b,
        output=output,
    )


def matmul(a: hl.Buffer, b: hl.Buffer) -> hl.Buffer:
    ty = a.type()

    type_info = (
        ty.code(),
        ty.bits(),
    )

    inner_size = a.width()
    output_width = b.width()
    output_height = a.height()
    size_threshold = 32

    print(f"{inner_size=} {output_width=} {output_height=}")

    is_big = (output_width > size_threshold) and (output_height > size_threshold) and (inner_size > size_threshold)

    if is_big:
        if type_info in _IMPLS_BIG:
            _IMPLS_BIG[type_info]
        else:
            _IMPLS_BIG[type_info] = _create_matmul_big(ty)
        impl = _IMPLS_BIG[type_info]
    else:
        if type_info in _IMPLS_SMALL:
            _IMPLS_SMALL[type_info]
        else:
            _IMPLS_SMALL[type_info] = _create_matmul_small(ty)
        impl = _IMPLS_SMALL[type_info]

    impl.a.set(a)
    impl.b.set(b)

    print(f"{a.width()=} {a.height()=}")
    print(f"{b.width()=} {b.height()=}")

    return impl.output.realize([b.width(), a.height()])
