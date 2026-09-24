// Constantes racine : la MVP partagée (colonnes, convention clip wgpu/Metal = D3D).
cbuffer Push : register(b0) { float4x4 mvp; };

struct VsOut {
    float4 pos : SV_Position;
    float3 color : COLOR;
};

VsOut vs_main(float3 pos : POSITION, float3 color : COLOR) {
    VsOut o;
    o.pos = mul(mvp, float4(pos, 1.0));
    o.color = color;
    return o;
}

float4 ps_main(VsOut i) : SV_Target {
    return float4(i.color, 1.0);
}
