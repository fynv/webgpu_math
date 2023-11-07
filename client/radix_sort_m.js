import { EngineContext } from "./EngineContext.js"

const WORKGROUP_SIZE = 64;
const WORKGROUP_SIZE_2x = WORKGROUP_SIZE * 2;

function condition(cond, a, b="")
{
    return cond? a: b;
}

function get_shader_radix_scan1(has_group_buf, workgroup_size = WORKGROUP_SIZE)
{
    let workgroup_size_2x = workgroup_size*2;

    return  `
struct Params
{
    count: i32,
    bit: u32
};

@group(0) @binding(0)
var<uniform> uParams: Params;

@group(0) @binding(1)
var<storage, read> bInput : array<i32>;

@group(0) @binding(2)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(3)
var<storage, read_write> bData2 : array<i32>;    

${condition(has_group_buf,`
@group(0) @binding(4)
var<storage, read_write> bGroup1 : array<i32>;

@group(0) @binding(5)
var<storage, read_write> bGroup2 : array<i32>;
`)}

var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData1);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let input = bInput[i];
        let pred = (input & (1 << uParams.bit)) != 0;
        s_buf1[threadIdx] = select(1,0,pred);
        s_buf2[threadIdx] = select(0,1,pred);
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        let input = bInput[i];
        let pred = (input & (1 << uParams.bit)) != 0;
        s_buf1[threadIdx + ${workgroup_size}] = select(1,0,pred);
        s_buf2[threadIdx + ${workgroup_size}] = select(0,1,pred);
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx];
        bData2[i] = s_buf2[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
    }

${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup1);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
    }
`)}
}
`;
}

function GetPipelineRadixScan1(has_group_buf)
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan1(has_group_buf) });
    let bindGroupLayouts = [has_group_buf ? engine_ctx.cache.bindGroupLayouts.radixScan1B : engine_ctx.cache.bindGroupLayouts.radixScan1A];
    const pipelineLayoutDesc = { bindGroupLayouts };
    let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

    return engine_ctx.device.createComputePipeline({
        layout,
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });
}

function get_shader_radix_scan2(has_group_buf, workgroup_size = WORKGROUP_SIZE)
{
    let workgroup_size_2x = workgroup_size*2;

    return  `    
@group(0) @binding(0)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(1)
var<storage, read_write> bData2 : array<i32>;    

${condition(has_group_buf,`
@group(0) @binding(2)
var<storage, read_write> bGroup1 : array<i32>;

@group(0) @binding(3)
var<storage, read_write> bGroup2 : array<i32>;
`)}

var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData1);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf1[threadIdx] = bData1[i];
        s_buf2[threadIdx] = bData2[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {        
        s_buf1[threadIdx + ${workgroup_size}] = bData1[i];
        s_buf2[threadIdx + ${workgroup_size}] = bData2[i];
    }

    workgroupBarrier();

    var half_size_group = 1u;
    var size_group = 2u;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<count)
        {
            s_buf1[tid] = s_buf1[gid*size_group + half_size_group -1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid*size_group + half_size_group -1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx];
        bData2[i] = s_buf2[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData1[i] = s_buf1[threadIdx + ${workgroup_size}];
        bData2[i] = s_buf2[threadIdx + ${workgroup_size}];
    }

${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup1);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup1[blockIdx] = s_buf1[${workgroup_size_2x} - 1];
        bGroup2[blockIdx] = s_buf2[${workgroup_size_2x} - 1];
    }
`)}

}
`
}

function GetPipelineRadixScan2(has_group_buf)
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan2(has_group_buf) });
    let bindGroupLayouts = [has_group_buf ? engine_ctx.cache.bindGroupLayouts.radixScan2B : engine_ctx.cache.bindGroupLayouts.radixScan2A];
    const pipelineLayoutDesc = { bindGroupLayouts };
    let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

    return engine_ctx.device.createComputePipeline({
        layout,
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });
}

function get_shader_radix_scan3(workgroup_size = WORKGROUP_SIZE)
{
    return  ` 
@group(0) @binding(0)
var<storage, read_write> bData1 : array<i32>;    

@group(0) @binding(1)
var<storage, read_write> bData2 : array<i32>;    

@group(0) @binding(2)
var<storage, read> bGroup1 : array<i32>;

@group(0) @binding(3)
var<storage, read> bGroup2 : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x + 2;    
    let count = arrayLength(&bData1);

    let add_idx = WorkgroupID.x / 2;
    let i = threadIdx + blockIdx*${workgroup_size};

    {
        let value = bData1[i];
        bData1[i] = value + bGroup1[add_idx];
    }

    {
        let value = bData2[i];
        bData2[i] = value + bGroup2[add_idx];
    }
}
`;
}

function GetPipelineRadixScan3()
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scan3() });
    let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScan3];
    const pipelineLayoutDesc = { bindGroupLayouts };
    let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

    return engine_ctx.device.createComputePipeline({
        layout,
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });

}

function get_shader_radix_scatter(workgroup_size = WORKGROUP_SIZE)
{
    return  `
@group(0) @binding(0)
var<uniform> uCount: i32;

@group(0) @binding(1)
var<storage, read> bInput : array<i32>;

@group(0) @binding(2)
var<storage, read> bIndices1 : array<i32>;

@group(0) @binding(3)
var<storage, read> bIndices2 : array<i32>;

@group(0) @binding(4)
var<storage, read_write> bOutput : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>)
{
    let idx = i32(GlobalInvocationID.x);
    if (idx>=uCount)
    {
        return;
    }

    let value = bInput[idx];
    if ((idx == 0 && bIndices1[idx]>0) || (idx > 0 && bIndices1[idx]>bIndices1[idx-1]))
    {
        bOutput[bIndices1[idx] - 1] = value;
    }
    else
    {
        let count0 = bIndices1[uCount -1];
        bOutput[count0 + bIndices2[idx] - 1] = value;
    }
}   
`    
}

function GetPipelineRadixScatter()
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader_radix_scatter() });
    let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScatter];
    const pipelineLayoutDesc = { bindGroupLayouts };
    let layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc);

    return engine_ctx.device.createComputePipeline({
        layout,
        compute: {
            module: shaderModule,
            entryPoint: 'main',
        },
    });
}

function getRandomInt(max) 
{
    return Math.floor(Math.random() * max);
}

export async function test()
{
    const engine_ctx = new EngineContext();
    await engine_ctx.initialize();

    let count = 64*64*64;
    let num_groups_radix_scan = Math.floor((count + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x);
    let max_value = 10000;
    let hInput = new Int32Array(count);
    for (let i=0; i<count; i++)
    {
        hInput[i] = getRandomInt(max_value);
    }

    let hReference = new Int32Array(count);
    hReference.set(hInput);
    hReference.sort();

    let buf_data = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    engine_ctx.queue.writeBuffer(buf_data, 0, hInput.buffer, hInput.byteOffset, hInput.byteLength);

    let buf_tmp = new Array(2);    
    buf_tmp[0] = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
    buf_tmp[1] = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);

    let buf_constant_radix_scan = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);

    let buffers_scan1 = [];
    let buffers_scan2 = [];
    let buf_sizes = [];
    let buf_size = count;
    while (buf_size>0)
    {
        let buf1 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);    
        let buf2 = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);    
        buffers_scan1.push(buf1);
        buffers_scan2.push(buf2);
        buf_sizes.push(buf_size);
        buf_size = Math.floor((buf_size + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x) - 1;
    }

    let buf_constant_radix_scatter = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);    
    let buf_download = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.COPY_DST| GPUBufferUsage.MAP_READ);

    let layout_entries_radix_scan1 = [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "uniform"
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
        {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
    ];

    if (buffers_scan1.length>1)
    {
        layout_entries_radix_scan1.push({
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });

        layout_entries_radix_scan1.push({
            binding: 5,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        });
    }

    let bindGroupLayoutRadixScan1 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan1 });

    if (buffers_scan1.length>1)
    {
        engine_ctx.cache.bindGroupLayouts.radixScan1B = bindGroupLayoutRadixScan1;    
    }
    else
    {
        engine_ctx.cache.bindGroupLayouts.radixScan1A = bindGroupLayoutRadixScan1;
    }

    let pipeline_radix_scan1 = GetPipelineRadixScan1(buffers_scan1.length>1);

    let layout_entries_radix_scan2 = [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
    ];

    let bindGroupLayoutRadixScan2A = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });
    engine_ctx.cache.bindGroupLayouts.radixScan2A = bindGroupLayoutRadixScan2A;

    layout_entries_radix_scan2.push({
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer:{
            type: "storage"
        }
    });

    layout_entries_radix_scan2.push({
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer:{
            type: "storage"
        }
    });

    let bindGroupLayoutRadixScan2B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan2 });
    engine_ctx.cache.bindGroupLayouts.radixScan2B = bindGroupLayoutRadixScan2B;

    let pipeline_radix_scan2A = GetPipelineRadixScan2(false);
    let pipeline_radix_scan2B = GetPipelineRadixScan2(true);

    let layout_entries_radix_scan3 = [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        },
        {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        }
    ];

    let bindGroupLayoutRadixScan3 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scan3 });
    engine_ctx.cache.bindGroupLayouts.radixScan3 = bindGroupLayoutRadixScan3;

    let pipeline_radix_scan3 = GetPipelineRadixScan3();

    let layout_entries_radix_scatter = [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "uniform"
            }
        },
        {
            binding: 1,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        },
        {
            binding: 2,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        },
        {
            binding: 3,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "read-only-storage"
            }
        },
        {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        },
    ];

    let bindGroupLayoutRadixScatter = engine_ctx.device.createBindGroupLayout({ entries: layout_entries_radix_scatter });
    engine_ctx.cache.bindGroupLayouts.radixScatter = bindGroupLayoutRadixScatter;

    let pipeline_radix_scatter = GetPipelineRadixScatter();

    let bind_group_radix_scan1 = new Array(2);
    let bind_group_radix_scatter = new Array(2);
    for (let i=0; i<2; i++)
    {
        let group_entries_radix_scan = [
            {
                binding: 0,
                resource:{
                    buffer: buf_constant_radix_scan
                }
            },
            {
                binding: 1,
                resource:{
                    buffer:  buf_tmp[i]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: buffers_scan1[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: buffers_scan2[0]
                }
            },
        ];

   
        if (buffers_scan1.length>1)
        {
            group_entries_radix_scan.push({
                binding: 4,
                resource:{
                    buffer: buffers_scan1[1]
                }
            });

            group_entries_radix_scan.push({
                binding: 5,
                resource:{
                    buffer: buffers_scan2[1]
                }
            });
        }  
        
        bind_group_radix_scan1[i] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan1, entries: group_entries_radix_scan});     

        let group_entries_radix_scatter = [
            {
                binding: 0,
                resource:{
                    buffer: buf_constant_radix_scatter
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: buf_tmp[i]            
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: buffers_scan1[0]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: buffers_scan2[0]
                }
            },
            {
                binding: 4,
                resource:{
                    buffer: buf_tmp[1-i]
                }
            },
        ];

        bind_group_radix_scatter[i] = engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScatter, entries: group_entries_radix_scatter});
    }

    let bind_group_radix_scan2 = [];
    for (let i=1; i<buffers_scan1.length; i++)
    {
        let group_entries_radix_scan = [            
            {
                binding: 0,
                resource:{
                    buffer: buffers_scan1[i]
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: buffers_scan2[i]
                }
            },
        ];

        if (i<buffers_scan1.length-1)
        {
            group_entries_radix_scan.push({
                binding: 2,
                resource:{
                    buffer: buffers_scan1[i+1]
                }
            });

            group_entries_radix_scan.push({
                binding: 3,
                resource:{
                    buffer: buffers_scan2[i+1]
                }
            });

            bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan2B, entries: group_entries_radix_scan}));
        }
        else
        {
            bind_group_radix_scan2.push(engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan2A, entries: group_entries_radix_scan}));
        }
    }

    let bind_group_radix_scan3 = [];
    for (let i=0; i < buffers_scan1.length - 1; i++)
    {
        let group_entries_radix_scan = [            
            {
                binding: 0,
                resource:{
                    buffer: buffers_scan1[i]
                }
            },
            {
                binding: 1,
                resource:{
                    buffer: buffers_scan2[i]
                }
            },
            {
                binding: 2,
                resource:{
                    buffer: buffers_scan1[i + 1]
                }
            },
            {
                binding: 3,
                resource:{
                    buffer: buffers_scan2[i + 1]
                }
            }
        ];
        bind_group_radix_scan3.push(engine_ctx.device.createBindGroup({ layout: bindGroupLayoutRadixScan3, entries: group_entries_radix_scan}));
    }


    let bits = 14;    

    {        
        let commandEncoder = engine_ctx.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buf_data, 0, buf_tmp[0], 0, count * 4);        
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);
    }

    for (let i=0; i<bits; i++)
    {
        {
            const uniform = new Int32Array(4);
            uniform[0] = count;
            uniform[1] = i;
            engine_ctx.queue.writeBuffer(buf_constant_radix_scan, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        {
            const uniform = new Int32Array(4);
            uniform[0] = count;
            engine_ctx.queue.writeBuffer(buf_constant_radix_scatter, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);
        }

        let commandEncoder = engine_ctx.device.createCommandEncoder();

        let j = i % 2;            

        {
            const passEncoder = commandEncoder.beginComputePass();
            {
                let num_groups = Math.floor((count + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
                passEncoder.setPipeline(pipeline_radix_scan1);
                passEncoder.setBindGroup(0, bind_group_radix_scan1[j]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            }

            for (let k = 0; k<bind_group_radix_scan2.length; k++)
            {
                let num_groups = Math.floor((buf_sizes[k+1] + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
                if (k<bind_group_radix_scan2.length - 1)
                {
                    passEncoder.setPipeline(pipeline_radix_scan2B);                
                }
                else
                {
                    passEncoder.setPipeline(pipeline_radix_scan2A);                
                }
                passEncoder.setBindGroup(0, bind_group_radix_scan2[k]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            }

            for (let k = bind_group_radix_scan3.length - 1; k>=0; k--)
            {
                let num_groups = Math.floor((buf_sizes[k] + WORKGROUP_SIZE - 1)/WORKGROUP_SIZE) - 2; 
                passEncoder.setPipeline(pipeline_radix_scan3);
                passEncoder.setBindGroup(0, bind_group_radix_scan3[k]);
                passEncoder.dispatchWorkgroups(num_groups, 1,1);
            }

            passEncoder.end();

        }
        

        {
            let num_groups = Math.floor((count + WORKGROUP_SIZE -1)/WORKGROUP_SIZE);
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(pipeline_radix_scatter);
            passEncoder.setBindGroup(0, bind_group_radix_scatter[j]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
            passEncoder.end();
        }


        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);
    }


    {
        let j = bits % 2;
        let commandEncoder = engine_ctx.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(buf_tmp[j], 0, buf_data, 0, count * 4);
        commandEncoder.copyBufferToBuffer(buf_data, 0, buf_download, 0, count * 4);
        let cmdBuf = commandEncoder.finish();
        engine_ctx.queue.submit([cmdBuf]);
    }

    let hOutput = new Int32Array(count);
    {   
        await buf_download.mapAsync(GPUMapMode.READ);
        let buf = buf_download.getMappedRange();
        hOutput.set(new Int32Array(buf));
        buf_download.unmap();
    }   

    console.log(hInput, hOutput, hReference);

    let count_unmatch = 0;
    for (let i=0; i<count; i++)
    {
        if (hOutput[i]!= hReference[i])
        {
            count_unmatch++;
        }
    }

    console.log(`count_unmatch: ${count_unmatch}`);

}

