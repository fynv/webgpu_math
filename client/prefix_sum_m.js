import { EngineContext } from "./EngineContext.js"

const WORKGROUP_SIZE = 64;
const WORKGROUP_SIZE_2x = WORKGROUP_SIZE * 2;

function condition(cond, a, b="")
{
    return cond? a: b;
}

function get_shader1(has_group_buf, workgroup_size = WORKGROUP_SIZE)
{
    let workgroup_size_2x = workgroup_size*2;

    return  `
@group(0) @binding(0)
var<storage, read_write> bData : array<i32>;    

${condition(has_group_buf,`
@group(0) @binding(1)
var<storage, read_write> bGroup : array<i32>;
`)}

var<workgroup> s_buf : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x;    
    let count = arrayLength(&bData);

    var i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx] = bData[i];
    }

    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        s_buf[threadIdx + ${workgroup_size}] = bData[i];
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
            s_buf[tid] = s_buf[gid*size_group + half_size_group -1] + s_buf[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    i = threadIdx + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData[i] = s_buf[threadIdx];
    }
    
    i = threadIdx + ${workgroup_size} + blockIdx*${workgroup_size_2x};
    if (i<count)
    {
        bData[i] = s_buf[threadIdx + ${workgroup_size}];
    }


${condition(has_group_buf,`        
    let count_group = arrayLength(&bGroup);
    if (threadIdx == 0 && blockIdx<count_group)
    {        
        bGroup[blockIdx] = s_buf[${workgroup_size_2x} - 1];
    }
`)}
}
`;
}

function GetPipeline1(has_group_buf)
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader1(has_group_buf) });
    let bindGroupLayouts = [has_group_buf ? engine_ctx.cache.bindGroupLayouts.compute1B : engine_ctx.cache.bindGroupLayouts.compute1A];
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

function get_shader2(workgroup_size = WORKGROUP_SIZE)
{
    return  ` 
@group(0) @binding(0)
var<storage, read_write> bData : array<i32>;    

@group(0) @binding(1)
var<storage, read> bGroup : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(
    @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
    @builtin(workgroup_id) WorkgroupID : vec3<u32>)
{
    let threadIdx = LocalInvocationID.x;
    let blockIdx = WorkgroupID.x + 2;    
    let count = arrayLength(&bData);

    let add_idx = WorkgroupID.x / 2;
    let i = threadIdx + blockIdx*${workgroup_size};
    let value = bData[i];
    bData[i] = value + bGroup[add_idx];
}
`;
}


function GetPipeline2()
{
    let shaderModule = engine_ctx.device.createShaderModule({ code: get_shader2() });
    let bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.compute2];
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

    let buffers = [];
    let buf_sizes = [];
    let buf_size = count;
    while (buf_size>0)
    {
        let buf = engine_ctx.createBuffer0(buf_size * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC);    
        buffers.push(buf);
        buf_sizes.push(buf_size);
        buf_size = Math.floor((buf_size + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x) - 1;

    }
    let buf_download = engine_ctx.createBuffer0(count * 4, GPUBufferUsage.COPY_DST| GPUBufferUsage.MAP_READ);

    engine_ctx.queue.writeBuffer(buffers[0], 0, hInput.buffer, hInput.byteOffset, hInput.byteLength);


    let layout_entries1 = [
        {
            binding: 0,
            visibility: GPUShaderStage.COMPUTE,
            buffer:{
                type: "storage"
            }
        }
    ];

    let bindGroupLayout1A = engine_ctx.device.createBindGroupLayout({ entries: layout_entries1 });
    engine_ctx.cache.bindGroupLayouts.compute1A = bindGroupLayout1A;

    layout_entries1.push({
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer:{
            type: "storage"
        }
    });

    let bindGroupLayout1B = engine_ctx.device.createBindGroupLayout({ entries: layout_entries1 });
    engine_ctx.cache.bindGroupLayouts.compute1B = bindGroupLayout1B;

    let layout_entries2 = [
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
                type: "read-only-storage"
            }
        }
    ];

    let bindGroupLayout2 = engine_ctx.device.createBindGroupLayout({ entries: layout_entries2 });
    engine_ctx.cache.bindGroupLayouts.compute2 = bindGroupLayout2;

    let groups1 = [];
    let groups2 = [];

    for (let i=0; i<buffers.length; i++)
    {
        if (i<buffers.length - 1)
        {
            let group_entries = [ 
                {
                    binding: 0,
                    resource:{
                        buffer: buffers[i]            
                    }
                },
                {
                    binding: 1,
                    resource:{
                        buffer: buffers[i+1]
                    }
                }       
            ];
            {
                let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayout1B, entries: group_entries});
                groups1.push(bind_group);
            }
            {
                let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayout2, entries: group_entries});
                groups2.push(bind_group);
            }
        }
        else if (buf_sizes[i]>1)
        {
            let group_entries = [ 
                {
                    binding: 0,
                    resource:{
                        buffer: buffers[i]            
                    }
                }              
            ];
            let bind_group = engine_ctx.device.createBindGroup({ layout: bindGroupLayout1A, entries: group_entries});
            groups1.push(bind_group);
        }
    }

    let pipeline1A = GetPipeline1(false);
    let pipeline1B = GetPipeline1(true);
    let pipeline2 =  GetPipeline2();

    let commandEncoder = engine_ctx.device.createCommandEncoder();    

    const passEncoder = commandEncoder.beginComputePass();

    for (let i=0; i<buffers.length; i++)
    {
        if (i<buffers.length - 1)
        {
            let num_groups = Math.floor((buf_sizes[i] + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
            passEncoder.setPipeline(pipeline1B);
            passEncoder.setBindGroup(0, groups1[i]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
        }
        else if (buf_sizes[i]>1)
        {
            let num_groups = Math.floor((buf_sizes[i] + WORKGROUP_SIZE_2x - 1)/WORKGROUP_SIZE_2x); 
            passEncoder.setPipeline(pipeline1A);
            passEncoder.setBindGroup(0, groups1[i]);
            passEncoder.dispatchWorkgroups(num_groups, 1,1); 
        }
    }   

    for (let i = buffers.length-2; i>=0; i--)
    {
        let num_groups = Math.floor((buf_sizes[i] + WORKGROUP_SIZE - 1)/WORKGROUP_SIZE) - 2; 
        passEncoder.setPipeline(pipeline2);
        passEncoder.setBindGroup(0, groups2[i]);
        passEncoder.dispatchWorkgroups(num_groups, 1,1);
    }
   
    passEncoder.end();

    commandEncoder.copyBufferToBuffer(buffers[0], 0, buf_download, 0, count * 4);

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

