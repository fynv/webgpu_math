import { EngineContext } from "./EngineContext.js"

const WORKGROUP_SIZE = 256;
const WORKGROUP_SIZE_2x = WORKGROUP_SIZE * 2;

function get_shader1(workgroup_size = WORKGROUP_SIZE)
{
    let workgroup_size_2x = workgroup_size*2;
    return  `
@group(0) @binding(0)
var<uniform> uCount: i32;

@group(0) @binding(1)
var<storage, read> bInput : array<i32>;

@group(0) @binding(2)
var<storage, read_write> bOutput : array<i32>;

@group(0) @binding(3)
var<storage, read_write> bWGCounter : atomic<i32>;

@group(0) @binding(4)
var<storage, read_write> bWGState : array<atomic<i32>>;

var<workgroup> s_workgroup_idx : i32;
var<workgroup> s_inclusive_prefix : i32;
var<workgroup> s_buf : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(local_invocation_id) LocalInvocationID : vec3<u32>)
{
    let threadIdx = i32(LocalInvocationID.x);
    if (threadIdx == 0)
    {
        s_workgroup_idx = atomicAdd(&bWGCounter, 1);
    }
    workgroupBarrier();

    let blockIdx = s_workgroup_idx; 

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        s_buf[threadIdx] = bInput[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        s_buf[threadIdx + ${workgroup_size}] = bInput[i];
    }

    workgroupBarrier();

    var half_size_group = 1;
    var size_group = 2;

    while(half_size_group <= ${workgroup_size})
    {
        let gid = threadIdx/half_size_group;
        let tid = gid*size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx*${workgroup_size_2x};
        if (i<uCount)
        {
            s_buf[tid] = s_buf[gid*size_group + half_size_group -1] + s_buf[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    s_inclusive_prefix = 0;
    if (threadIdx == 0)
    {
        atomicStore(&bWGState[blockIdx*3 + 1],  s_buf[${workgroup_size_2x} - 1]);
        atomicStore(&bWGState[blockIdx*3], 1);

        var j = blockIdx;
        while(j>0)
        {
            j--;    
            var state = 0;
            while(true)
            {
                state = atomicLoad(&bWGState[j*3]);                
                if (state>0) 
                {                   
                    break;
                }
            }
            
            if (state==2)
            {
                s_inclusive_prefix+= atomicLoad(&bWGState[j*3 + 2]);
                break;
            }
            else
            {
                s_inclusive_prefix+= atomicLoad(&bWGState[j*3 + 1]);
            }
        }         

        atomicStore(&bWGState[blockIdx*3 + 2], s_buf[${workgroup_size_2x} - 1] + s_inclusive_prefix);
        atomicStore(&bWGState[blockIdx*3], 2);
    }
    workgroupBarrier();

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        bOutput[i] = s_buf[threadIdx] + s_inclusive_prefix;
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<uCount)
    {
        bOutput[i] = s_buf[threadIdx + ${workgroup_size}] + s_inclusive_prefix;
    }
}
`;
    
}

function GetPipeline1()
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader1() });
    let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.compute1];
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

    let count = 65536;
    let num_groups = Math.floor((count + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x);
    let max_value = 100;
    let hInput = new Int32Array(count);
    for (let i=0; i<count; i++)
    {
        hInput[i] = getRandomInt(max_value);
    }
    
    let hReference = new Int32Array(count);

    let acc = 0;
    for (let i=0; i<count; i++)
    {
        acc+= hInput[i];
        hReference[i] = acc;
    }

    let buf_constant1 = engine_ctx.createBuffer0(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    let buf_input = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let buf_output = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
    let buf_download = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.COPY_DST| GPUBufferUsage.MAP_READ);
    let buf_workgroup_counter = engine_ctx.createBuffer0(4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let buf_workgroup_state = engine_ctx.createBuffer0(num_groups * 3 *4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);

    engine_ctx.queue.writeBuffer(buf_input, 0, hInput.buffer, hInput.byteOffset, hInput.byteLength);

    let layout_entries1 = [
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
        {
            binding: 4,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        }
    ];

    let bindGroupLayout1 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries1 });
    engine_ctx.cache.bindGroupLayouts.compute1 = bindGroupLayout1;
    
    let group_entries1 = [
        {
            binding: 0,
            resource:{
                buffer: buf_constant1
            }
        },
        {
            binding: 1,
            resource:{
                buffer: buf_input            
            }
        },
        {
            binding: 2,
            resource:{
                buffer: buf_output            
            }
        },
        {
            binding: 3,
            resource:{
                buffer: buf_workgroup_counter            
            }
        },
        {
            binding: 4,
            resource:{
                buffer: buf_workgroup_state            
            }
        }
    ];

    let bind_group1 = engine_ctx.device.createBindGroup({ layout: bindGroupLayout1, entries: group_entries1});

    const uniform = new Int32Array(4);
    uniform[0] = count;
    engine_ctx.queue.writeBuffer(buf_constant1, 0, uniform.buffer, uniform.byteOffset, uniform.byteLength);

    const group_count = new Int32Array(1, 0);    
    engine_ctx.queue.writeBuffer(buf_workgroup_counter, 0, group_count.buffer, group_count.byteOffset, group_count.byteLength);

    const group_state = new Int32Array(num_groups*3, 0);
    engine_ctx.queue.writeBuffer(buf_workgroup_state, 0, group_state.buffer, group_state.byteOffset, group_state.byteLength);

    let pipeline1 = GetPipeline1();

    let commandEncoder = engine_ctx.device.createCommandEncoder();    

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(pipeline1);
    passEncoder.setBindGroup(0, bind_group1);
    passEncoder.dispatchWorkgroups(num_groups, 1,1); 
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(buf_output, 0, buf_download, 0, count * 4);

    let cmdBuf = commandEncoder.finish();
    engine_ctx.queue.submit([cmdBuf]);

    let hOutput = new Int32Array(count);
    await buf_download.mapAsync(GPUMapMode.READ);
    let buf = buf_download.getMappedRange();
    hOutput.set(new Int32Array(buf));
    buf_download.unmap();

    console.log(hInput, hOutput, hReference)

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