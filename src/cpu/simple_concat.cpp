/*******************************************************************************
* Copyright 2017-2021 Intel Corporation
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

#include <cstring>

#include "common/dnnl_thread.hpp"

#include "cpu/simple_concat.hpp"

#include <iostream>

#define PRAGMA_EXEC
// #define COPY_EXEC
// #define NEW_PARALLEL

namespace dnnl {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

template <data_type_t data_type>
status_t simple_concat_t<data_type>::execute(const exec_ctx_t &ctx) const {
    auto scratchpad = ctx.get_scratchpad_grantor();
    auto iptrs = scratchpad.template get<const data_t *>(key_concat_iptrs);
    auto optrs = scratchpad.template get<data_t *>(key_concat_optrs);
    auto nelems_to_copy = scratchpad.template get<dim_t>(key_concat_nelems);
    auto is = scratchpad.template get<strides_t>(key_concat_istrides);

    const int num_arrs = pd()->n_inputs();
    const int *perm = pd()->perm_, *iperm = pd()->iperm_;
    const int concat_dim = pd()->concat_dim();
    auto o_base_ptr = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);
    if (o_base_ptr == nullptr) return status::success;
    std::cout << "out ptr simple_concat_t: " << o_base_ptr << std::endl;
    for (int a = 0; a < num_arrs; ++a) {
        const memory_desc_wrapper i_d(pd()->src_md(a));
        const memory_desc_wrapper o_d(pd()->src_image_md(a));
        const auto iptr = CTX_IN_MEM(const data_t *, DNNL_ARG_MULTIPLE_SRC + a);
        std::cout << a << " in ptr simple_concat_t: " << iptr << std::endl;
        if (iptr == nullptr) {
            iptrs[a] = nullptr;
            nelems_to_copy[a] = 0;
            continue;
        }
        iptrs[a] = iptr + i_d.blk_off(0);
        optrs[a] = o_base_ptr + o_d.blk_off(0);
        nelems_to_copy[a] = pd()->nelems_to_concat(i_d);
        for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
            if (i < perm[concat_dim])
                is[a][i] = size_t(i_d.blocking_desc().strides[iperm[i]]);
            else
                is[a][i] = 0;
        }
        std::cout << "in offset: " << i_d.blk_off(0) << " img offset: " << o_d.blk_off(0) << std::endl;
        std::cout << "istrides" << std::endl;
        for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
            std::cout << is[a][i] << " ";
        }
        std::cout << std::endl;
    }

    const memory_desc_wrapper o_d(pd()->dst_md(0));

    strides_t os = {0};
    bool has_outer_loop = false;
    for (int i = 0; i < perm[concat_dim]; i++) {
        os[i] = o_d.blocking_desc().strides[iperm[i]];
        // CAVEAT: if this impl supports not matching stag and dtag, strides
        // should be taken into account for this condition.
        if (o_d.padded_dims()[iperm[i]] != 1) has_outer_loop = true;
    }
    std::cout << "ostrides" << std::endl;
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        std::cout << os[i] << " ";
    }
    std::cout << std::endl;

    // Applies when concat axis is the outermost dimension, e.g. concat_axis = 0
    // or concat_axis = 1, and dims[0] = 1;
    if (!has_outer_loop) {
        for (int a = 0; a < num_arrs; ++a) {
            const data_t *i = &iptrs[a][0];
            data_t *o = &optrs[a][0];
            #ifdef NEW_PARALLEL
            parallel_nd(nelems_to_copy[a], [&](dim_t e) { o[e] = i[e]; });
            #else
            parallel_nd_legacy(nelems_to_copy[a], [&](dim_t e) { o[e] = i[e]; });
            #endif
        }
        return status::success;
    }

    dims_t phys_dims;
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        if (i < perm[concat_dim])
            phys_dims[i]
                    = o_d.padded_dims()[iperm[i]] / pd()->blocks_[iperm[i]];
        else
            phys_dims[i] = 1;
    }

    std::cout << "phys_dims" << std::endl;
    for (int i = 0; i < DNNL_MAX_NDIMS; i++) {
        std::cout << phys_dims[i] << " ";
    }
    std::cout << std::endl;

    #ifdef NEW_PARALLEL
    std::cout << "NEW_PARALLEL" << std::endl;
    parallel_nd(phys_dims[0], phys_dims[1], phys_dims[2], phys_dims[3],
    #else
    std::cout << "LEGACY_PARALLEL" << std::endl;
    parallel_nd_legacy(phys_dims[0], phys_dims[1], phys_dims[2], phys_dims[3],
    #endif
            phys_dims[4], num_arrs,
            [&](dim_t n0, dim_t n1, dim_t n2, dim_t n3, dim_t n4, dim_t a) {
                std::cout << "================= concat parallel start =================" << std::endl;
                std::cout << n0 << " " << n1 << " " << n2 << " " << n3 << " " << n4 << " num in: " << a << std::endl;
                // check if zero memory
                if (iptrs[a] == nullptr) return;

                // XXX: this code may access uninitialized values in is[*][0-4] --
                // that's why we have to set them to zero although this is
                // probably benign
                size_t in_off = is[a][0] * n0 + is[a][1] * n1 + is[a][2] * n2
                        + is[a][3] * n3 + is[a][4] * n4;
                size_t out_off = os[0] * n0 + os[1] * n1 + os[2] * n2
                        + os[3] * n3 + os[4] * n4;
                const data_t *i = &iptrs[a][in_off];
                data_t *o = &optrs[a][out_off];
                std::cout << "in_off: " << in_off << " out_off: " << out_off << " nelems_to_copy: " << nelems_to_copy[a] << std::endl;
                std::cout << "ptr_diff in: " << (o - o_base_ptr) << " ptr_diff out: " << (i - CTX_IN_MEM(const data_t *, DNNL_ARG_MULTIPLE_SRC + a)) << std::endl;
                #ifdef PRAGMA_EXEC
                std::cout << "PRAGMA_EXEC" << std::endl;
                PRAGMA_OMP_SIMD()
                for (dim_t e = 0; e < nelems_to_copy[a]; ++e) {
                    o[e] = i[e];
                }
                #elif defined(COPY_EXEC)
                std::cout << "COPY_EXEC" << std::endl;
                for (dim_t e = 0; e < nelems_to_copy[a]; ++e) {
                    o[e] = i[e];
                }
                #else
                std::cout << "MEMCPY" << std::endl;
                std::memcpy(o, i, nelems_to_copy[a] * sizeof(data_t));
                #endif

                std::cout << "DATA: " << o[nelems_to_copy[a] - 1] << " " <<  i[nelems_to_copy[a] - 1] << std::endl;
                std::cout << "================= concat parallel end =================" << std::endl;
            });

    return status::success;
}

template struct simple_concat_t<data_type::f32>;
template struct simple_concat_t<data_type::u8>;
template struct simple_concat_t<data_type::s8>;
template struct simple_concat_t<data_type::s32>;
template struct simple_concat_t<data_type::bf16>;

} // namespace cpu
} // namespace impl
} // namespace dnnl
