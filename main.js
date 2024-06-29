// WebGPU Simple Textured Quad MipFilter
// from https://webgpufundamentals.org/webgpu/webgpu-simple-textured-quad-mipmapfilter.html

// see https://webgpufundamentals.org/webgpu/lessons/webgpu-utils.html#wgpu-matrix
//import {mat4} from 'https://webgpufundamentals.org/3rdparty/wgpu-matrix.module.js';
const clearColor = { r: 0.0, g: 0., b: 0.0, a: 1.0 };
const rainbow = [
  [1.0, 0.0, 0.0],  // Red
  [1.0, 0.65, 0.0], // Orange
  [1.0, 1.0, 0.0],  // Yellow
  [0.0, 1.0, 0.0],  // Green
  [0.0, 0.0, 1.0],  // Blue
  [0.3, 0.0, 0.5],  // Indigo
  [0.5, 0.0, 1.0]   // Violet/Purple
];
const numParticles = 256;
const particleInstanceByteSize =
  4 * 4 + // position
  4 * 4; // color
const particlePositionOffset = 0;
const particleColorOffset = 4 * 4;

// Shader 정의
const shaders = `
struct VertexInput {
  @location(0) position : vec4f,
  @location(1) color : vec4f,
  @location(2) quad_pos : vec2f, // -1..+1
}
struct VertexOut {
  @builtin(position) position : vec4f,
  @location(0) color : vec4f,
  @location(1) quad_pos : vec2f, // -1..+1
}

@vertex
fn vertex_main(in : VertexInput) -> VertexOut
{
  var output : VertexOut;
  output.position.x = in.position.x + in.quad_pos.x * 0.03;
  output.position.y = in.position.y + in.quad_pos.y * 0.03;
  output.position.z = in.position.z;
  output.position.w = in.position.w;
  output.color = in.color;
  output.quad_pos = in.quad_pos;
  return output;
}

@fragment
fn fragment_main(fragData: VertexOut) -> @location(0) vec4f
{
  var OutData = fragData.color;
  OutData.a = fragData.color.a * max(1.0 - length(fragData.quad_pos), 0.0);
  return OutData;
}
`;

function getRandomInt(min, max) {
  min = Math.ceil(min);
  max = Math.floor(max);
  return Math.floor(Math.random() * (max - min + 1)) + min;
}
function getRandomfloat(min, max) {
  return Math.random() * (max - min + 1) + min;
}
function createParticles(){
  const particles = [];
  for(let i = 0; i<numParticles; i++)
    {
      let x = getRandomfloat(-1.0, 1.0);
      let y = getRandomfloat(-1.0, 1.0);
      let z = 0.0;
      let w = 1.0;
      
      let r = rainbow[getRandomInt(0, 6)][0];
      let g = rainbow[getRandomInt(0, 6)][1];
      let b = rainbow[getRandomInt(0, 6)][2];
      let a = 1.0;
      particles.push(x,y,z,w,
        r,g,b,a
      );
    }
    // 결국 동적 할당으로 이리 빼줘야되는구나,,
    return new Float32Array(particles);
}
const particleData = createParticles();

async function copyTexture(device, sourceTexture, destinationTexture, width, height) {
  const commandEncoder = device.createCommandEncoder();

  commandEncoder.copyTextureToTexture(
      { texture: sourceTexture }, // Source texture
      { texture: destinationTexture }, // Destination texture
      [width, height, 1] // Copy size
  );

  const commandBuffer = commandEncoder.finish();
  device.queue.submit([commandBuffer]);
}

var device;
var context;
var particlesBuffer;
var quadVertexBuffer; 
var densityTexture;
var densityTmpTexture;
var sampler;
async function init() {

  try{
  // 1: request adapter and device
  if (!navigator.gpu) {
    throw Error('WebGPU not supported.');
  }
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw Error('Couldn\'t request WebGPU adapter.');
  }
  device = await adapter.requestDevice();
  if(!device)
    {
      throw new Error('Failed to get GPU device.');
    }
    device.lost.then((info) => {
      console.error('Device lost:', info.message);
  });

  //console.log(firstMessage);
  // 3: Get reference to the canvas to render on
  const canvas = document.querySelector('#gpuCanvas');
  context = canvas.getContext('webgpu');

  context.configure({
    device: device,
    format: 'rgba16float',//navigator.gpu.getPreferredCanvasFormat(),
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST,
    alphaMode: 'premultiplied'
  });

  particlesBuffer = device.createBuffer({
    size: numParticles * particleInstanceByteSize,
    usage: GPUBufferUsage.VERTEX | GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });
  new Float32Array(particlesBuffer.getMappedRange()).set(particleData);
  particlesBuffer.unmap();


      // 버텍스 버퍼 초기화
  quadVertexBuffer = device.createBuffer({
      size: 6 * 2 * 4, // 6x vec2f
      usage: GPUBufferUsage.VERTEX,
      mappedAtCreation: true,
    });
    const vertexData = [
      -1.0, -1.0, +1.0, -1.0, -1.0, +1.0, -1.0, +1.0, +1.0, -1.0, +1.0, +1.0,
    ];
    new Float32Array(quadVertexBuffer.getMappedRange()).set(vertexData);
    quadVertexBuffer.unmap();
    var a = canvas.width; 
    var b = canvas.height;
    densityTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: context.getCurrentTexture().format,
      loadValue: clearColor, // 초기 색상 값 (검정색: RGBA가 모두 0)
      usage: GPUTextureUsage.COPY_SRC|
      GPUTextureUsage.RENDER_ATTACHMENT|
      GPUTextureUsage.STORAGE_BINDING|
      GPUTextureUsage.TEXTURE_BINDING
    });
    densityTmpTexture = device.createTexture({
      size: [canvas.width, canvas.height],
      format: context.getCurrentTexture().format,
      loadValue: clearColor, // 초기 색상 값 (검정색: RGBA가 모두 0)
      usage: GPUTextureUsage.COPY_DST|
      GPUTextureUsage.RENDER_ATTACHMENT|
      GPUTextureUsage.STORAGE_BINDING|
      GPUTextureUsage.TEXTURE_BINDING
    });
    sampler = device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }catch(error){
    console.error('WebGPU initialization failed:', error);
  }


}

function dissipateDensity(device, context)
{
  // Copy
  copyTexture(device, densityTexture, densityTmpTexture, densityTexture.width, densityTexture.height);

  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        sampler: { type: 'filtering' }, // 샘플러 타입 설정
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture:{
          access: 'read-only',
          format: 'rgba16float'
        }
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture:{
          access: 'write-only',
          format: 'rgba16float'
        }
      },
    ],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: sampler
      },
      {
        binding: 1,
        resource: densityTmpTexture.createView({
          format: 'rgba16float',
          dimension: '2d',
        })
      },
      {
        binding: 2,
        resource: densityTexture.createView({
          format: 'rgba16float',
          dimension: '2d',
        })
      },
    ],
    label: 'RenderPassLabel',
  });
 // 
 //textureStore(outputTex, vec2<u32>(u32(x), u32(y)), vec4<f32>(color, 1.0));
 //textureStore(outputTex, vec2<u32>(global_id.x, global_id.y), vec4<f32>(color, 1.0));
        //var x : f32 = f32(global_id.x);
        //var y : f32 = f32(global_id.y);
        //let coordX :f32 = x/800.0; //f32(textureDimensions(inputTex).x);
        //let coordY :f32 = y/600.0; //f32(textureDimensions(inputTex).y);
  const shaderModule = device.createShaderModule({
    code: `
      @group(0) @binding(0) var samp : sampler;
      @group(0) @binding(1) var inputTex : texture_storage_2d<rgba16float, read>;
      @group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>; 
      @compute @workgroup_size(8, 8, 1)
      fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {

        const dissipation = 0.1;
        var color = textureLoad(inputTex,  vec2<u32>(global_id.x, global_id.y));
        
        if(length(color) >0.0)
        {
          var newColor = color.xyz - color.xyz*dissipation;
          textureStore(outputTex, vec2<u32>(global_id.x, global_id.y), vec4(newColor, color.a));
        }
        else
        {
          //textureStore(outputTex, vec2<u32>(global_id.x, global_id.y), color);
        }
      }
        
    `
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [bindGroupLayout]
    }),
    compute: {
      module: shaderModule,
      entryPoint: "main"
    }
  });

  //let velocity = vec4f(-particle.position.y, particle.position.x, 0.0, 0.0);
  //datas.particles[index].position = particle.position + velocity*dt;
    // Compute shader code
    /*
    if (length(densityField[dtID.xy]) > 0.0)
    {
      densityField[dtID.xy] = densityField[dtID.xy] - dissipation * densityField[dtID.xy];
    }*/

      // Commands submission
    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(Math.ceil(densityTexture.width/8.0), Math.ceil(densityTexture.height/8.0));
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

function advectParticle(device, context)
{
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage"
        }
      },
    ],
  });
  const bindGroup = device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      {
        binding: 0,
        resource: {
          buffer: particlesBuffer,
          offset : 0,
          size : numParticles * particleInstanceByteSize,
        }
      }
    ]
  });
  //let velocity = vec4f(-particle.position.y, particle.position.x, 0.0, 0.0);
  //datas.particles[index].position = particle.position + velocity*dt;
    // Compute shader code
    const shaderModule = device.createShaderModule({
      code: `
        struct Particle {
          position : vec4f,
          color : vec4f,
        }
          struct Particles{
            particles : array<Particle>
          }
        @group(0) @binding(0) var<storage, read_write> datas : Particles;
  
        @compute @workgroup_size(256)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
          let index : u32 = global_id.x;
          var particle = datas.particles[index];
          const dt = 1/120.0;
          let velocity = vec4f(-particle.position.y, particle.position.x, 0.0, 0.0);
          datas.particles[index].position = particle.position +velocity * dt;
        }
      `
    });

    const computePipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({
        bindGroupLayouts: [bindGroupLayout]
      }),
      compute: {
        module: shaderModule,
        entryPoint: "main"
      }
    });

      // Commands submission
    const commandEncoder = device.createCommandEncoder();

    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(1, 1);
    passEncoder.end();
    device.queue.submit([commandEncoder.finish()]);
}

//////////// Draw
async function draw(device, context)
{
try{
     // 2: Create a shader module from the shaders template literal
     const shaderModule = device.createShaderModule({
      code: shaders
    });
  
   // 쉐이더 모듈 컴파일 정보 가져오기
     const compilationInfo = await shaderModule.getCompilationInfo();
          if (compilationInfo.messages.length > 0) {
              console.warn('Shader compilation messages:');
              compilationInfo.messages.forEach((msg) => {
                  console.warn(`${msg.message} (Type: ${msg.type}, Line: ${msg.lineNum}, Position: ${msg.linePos})`);
              });
          }
  // Copy the vertex data over to the GPUBuffer using the writeBuffer() utility function
  // device.queue.writeBuffer(particlesBuffer, 0, particleData, 0, particleData.length); // Map과 큰 차이가,,? 비동기?
  // 5: Create a GPUVertexBufferLayout and GPURenderPipelineDescriptor to provide a definition of our render pipline
  // 각 vertex마다, instance 마다 이렇게 보낼 수 있는 꿀팁이..
  const vertexBuffers = [{
    arrayStride: particleInstanceByteSize,
    stepMode: 'instance',
    attributes: [{
      shaderLocation: 0, // position
      offset: particlePositionOffset,
      format: 'float32x4'
    }, {
      shaderLocation: 1, // color
      offset: particleColorOffset,
      format: 'float32x4'
    }],
  },
  {
    // quad vertex buffer
    arrayStride: 2 * 4, // vec2f
    stepMode: 'vertex',
    attributes: [
      {
        // vertex positions
        shaderLocation: 2,
        offset: 0,
        format: 'float32x2',
      },
    ],
  }
];
const blendState = {
  color: {
    srcFactor: 'one',
    dstFactor: 'one',
    operation: 'add',
  },
  alpha: {
    srcFactor: 'one',
    dstFactor: 'one',
    operation: 'add',
  },
};
// Pipeline 생성, 이때 셰이더랑 버텍스 버퍼 연결
  const pipelineDescriptor = {
    vertex: {
      module: shaderModule,
      entryPoint: 'vertex_main',
      buffers: vertexBuffers
    },
    fragment: {
      module: shaderModule,
      entryPoint: 'fragment_main',
      targets: [{
        format: 'rgba16float',//navigator.gpu.getPreferredCanvasFormat()
        blend : blendState,
      }]
    },
    primitive: {
      topology: 'triangle-list'
    },
    layout: 'auto'
  };
  const renderPipeline = device.createRenderPipeline(pipelineDescriptor);

  // 7: Create GPUCommandEncoder to issue commands to the GPU
  // Note: render pass descriptor, command encoder, etc. are destroyed after use, fresh one needed for each frame.
  const commandEncoder = device.createCommandEncoder();
  // 8: Create GPURenderPassDescriptor to tell WebGPU which texture to draw into, then initiate render pass

  const renderPassDescriptor = {
    colorAttachments: [{
      clearValue: clearColor,
      loadOp: 'load',//load
      storeOp: 'store',
      view: densityTexture.createView()
    }]
  };

  const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
    
  // 9: Draw the triangle
  passEncoder.setPipeline(renderPipeline);
  passEncoder.setVertexBuffer(0, particlesBuffer);
  passEncoder.setVertexBuffer(1, quadVertexBuffer);
  passEncoder.draw(6, numParticles,0, 0);

  // End the render pass
  passEncoder.end();

  // 10: End frame by passing array of command buffers to command queue for execution
  device.queue.submit([commandEncoder.finish()]);
  // 현재 화면에 복사
  var swapChainTexture = context.getCurrentTexture();
  copyTexture(device,densityTexture ,swapChainTexture, swapChainTexture.width, swapChainTexture.height);
  } catch(error){
    console.error('WebGPU Draw failed:', error);
  }
}

await init();
while(true)
  {
    dissipateDensity(device, context);
    advectParticle(device, context);
    await draw(device, context);
  }


