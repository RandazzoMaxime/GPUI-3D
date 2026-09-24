#include <metal_stdlib>
using namespace metal;

struct Vertex { packed_float3 pos; packed_float3 color; };
struct VsOut { float4 pos [[position]]; float3 color; };

vertex VsOut vs_main(uint vid [[vertex_id]],
                     const device Vertex* vertices [[buffer(0)]],
                     constant float4x4& mvp [[buffer(1)]]) {
    Vertex v = vertices[vid];
    return VsOut { mvp * float4(float3(v.pos), 1.0), float3(v.color) };
}

fragment float4 fs_main(VsOut in [[stage_in]]) {
    return float4(in.color, 1.0);
}
