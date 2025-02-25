/*******************************************************************************
* Copyright 2019-2021 Intel Corporation
* Copyright 2020 FUJITSU LIMITED
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "cpu/cpu_engine.hpp"

#include "cpu/nchw_pooling.hpp"
#include "cpu/nhwc_pooling.hpp"
#include "cpu/ref_pooling.hpp"

#if DNNL_X64
#include "cpu/x64/jit_uni_i8i8_pooling.hpp"
#include "cpu/x64/jit_uni_pooling.hpp"
using namespace dnnl::impl::cpu::x64;
#elif DNNL_AARCH64
#include "cpu/aarch64/jit_uni_i8i8_pooling.hpp"
#include "cpu/aarch64/jit_uni_pooling.hpp"
using namespace dnnl::impl::cpu::aarch64;
#endif

namespace dnnl {
namespace impl {
namespace cpu {

namespace {
using namespace dnnl::impl::data_type;

// clang-format off
const impl_list_item_t impl_list[] = {
        /* fp */
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, avx512_core, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, avx512_core, bf16))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, avx512_core, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, avx512_core, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, avx512_common, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, avx512_common, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, avx2, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, avx2, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, avx, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, avx, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_pooling_fwd_t, sse41, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE_X64(jit_uni_pooling_bwd_t, sse41, f32))
        REG_POOLING_P_FWD(CPU_INSTANCE_AARCH64(jit_uni_pooling_fwd_t, sve_512, f32))
        REG_POOLING_P_BWD(CPU_INSTANCE_AARCH64(jit_uni_pooling_bwd_t, sve_512, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(nchw_pooling_fwd_t, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE(nchw_pooling_bwd_t, bf16))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(nchw_pooling_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE(nchw_pooling_bwd_t, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(nhwc_pooling_fwd_t, bf16))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE(nhwc_pooling_bwd_t, bf16))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(nhwc_pooling_fwd_t, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_BWD(CPU_INSTANCE(nhwc_pooling_bwd_t, f32))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, f32, f32, f32))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, bf16, bf16, f32))
        REG_POOLING_P_BWD(CPU_INSTANCE(ref_pooling_bwd_t, f32))
        REG_POOLING_P_BWD(CPU_INSTANCE(ref_pooling_bwd_t, bf16))
#endif
        /* int */
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t, avx512_core))
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t, avx2))
        REG_POOLING_P_FWD(CPU_INSTANCE_X64(jit_uni_i8i8_pooling_fwd_t, sse41))
#ifdef ENABLE_UNUSED_PRIM
        REG_POOLING_P_FWD(CPU_INSTANCE_AARCH64(jit_uni_i8i8_pooling_fwd_t, sve_512))
#endif
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, s32, s32, s32))
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, s8, s8, s32))
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, s8, f32, f32))
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, u8, u8, s32))
        REG_POOLING_P_FWD(CPU_INSTANCE(ref_pooling_fwd_t, u8, f32, f32))
#ifdef ENABLE_UNUSED_PRIM
#endif
        /* eol */
        nullptr,
};
// clang-format on
} // namespace

const impl_list_item_t *get_pooling_v2_impl_list(
        const pooling_v2_desc_t *desc) {
    UNUSED(desc);
    return impl_list;
}

} // namespace cpu
} // namespace impl
} // namespace dnnl
