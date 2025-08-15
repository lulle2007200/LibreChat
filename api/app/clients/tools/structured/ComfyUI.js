// Generates image using stable diffusion webui's api (automatic1111)
const fs = require('fs');
const { z } = require('zod');
const path = require('path');
const axios = require('axios');
const sharp = require('sharp');
const { v4: uuidv4 } = require('uuid');
const { Tool } = require('@langchain/core/tools');
const { FileContext, ContentTypes } = require('librechat-data-provider');
const paths = require('~/config/paths');
const { logger } = require('~/config');

const displayMessage =
  'Stable Diffusion displayed an image. All generated images are already plainly visible, so don\'t repeat the descriptions in detail. Do not list download links as they are available in the UI already. The user may download the images by clicking on them, but do not mention anything about downloading to the user.';

class ComfyUiBase extends Tool {
  constructor(fields) {
    super();
    /** @type {string} User ID */
    this.userId = fields.userId;
    /** @type {Express.Request | undefined} Express Request object, only provided by ToolService */
    this.req = fields.req;
    /** @type {boolean} Used to initialize the Tool without necessary variables. */
    this.override = fields.override ?? false;
    /** @type {boolean} Necessary for output to contain all image metadata. */
    this.returnMetadata = fields.returnMetadata ?? false;
    /** @type {boolean} */
    this.isAgent = fields.isAgent;
    if (fields.uploadImageBuffer) {
      /** @type {uploadImageBuffer} Necessary for output to contain all image metadata. */
      this.uploadImageBuffer = fields.uploadImageBuffer.bind(this);
    }


    this.url            = fields.COMFYUI_URL      || this.getServerURL();
    
    // If specified, included as bearer in any request
    this.api_key        = fields.COMFYUI_API_KEY  || this.getAPIKey();

    // required. Should be a valid ComfyUI API workflow
    this.workflowStr    = fields.COMFYUI_WORKFLOW || this.getWorkflow();
    this.workflow       = this.parseWorkflow();

    // optional. Can be used to map properties to workflow nodes
    // e.g. {"widht": "1", "height": "2", "steps": "3", "samplerName": "4", "sampler": "5", "positive": "6", "negative": "7", "model": "8"}
    this.nodeMapStr     = fields.COMFYUI_NODE_MAP || this.getNodeMap();
    this.nodeMap        = this.parseNodeMap();


    // required
    this.positiveNodeId    = this.getPositivePromptNodeId();

    // optional
    this.samplerNodeId     = this.getSamplerNodeId();
    this.modelNodeId       = this.getModelNodeId();
    this.negativeNodeId    = this.getNegativePromptNodeId();
    this.seedNodeId        = this.getSeedNodeId();
    this.samplerNameNodeId = this.getSamplerNameNodeId();
    this.stepsNodeId       = this.getStepsNodeId();
    this.widthNodeId       = this.getWidthNodeId();
    this.heightNodeId      = this.getHeightNodeId();
  }

  parseNodeMap() {
    if(this.override) {
      return {};
    }

    return JSON.parse(this.nodeMapStr);
  }

  parseWorkflow() {
    if(this.override) {
      return {}
    }

    return JSON.parse(this.workflowStr);
  }

  checkProperty(obj, name) {
    if(obj && Object.prototype.hasOwnProperty.call(obj, name) && (typeof(obj[name]) !== "object" || obj[name] === null)) {
      return true;
    }
    return false;
  }

  getSizeNodeId(name){
    let sizeNodeId = this.nodeMap?.[name] || '';
    if(!sizeNodeId) {
      const samplerNodeId = this.getSamplerNodeId();
      sizeNodeId    = this.workflow?.[samplerNodeId]?.inputs?.latent_image?.[0] || '';
    }

    if(!this.checkProperty(this.workflow?.[sizeNodeId]?.inputs, name)){
      logger.info(`${name} node doesnt have ${name} input`);
      sizeNodeId = '';
    }

    return sizeNodeId;
  }

  getWidthNodeId() {
    return this.getSizeNodeId("width");
  }

  getHeightNodeId() {
    return this.getSizeNodeId("height");
  }

  getStepsNodeId() {
    let stepsNodeId = this.nodeMap?.steps || '';

    if(!stepsNodeId) {
      const samplerNodeId = this.getSamplerNodeId();
      stepsNodeId         = samplerNodeId;
      if (this.workflow?.[samplerNodeId]?.class_type === "CustomSampler") {
        stepsNodeId = this.workflow?.[samplerNodeId]?.inputs?.sigmas?.[0] ||'';
      }
    }

    if (!this.checkProperty(this.workflow[stepsNodeId].inputs, "steps")){
      stepsNodeId = '';
      logger.info("steps node doesn't have steps input.");
    }

    return stepsNodeId;
  }

  getNodeMap() {
    let nodeMap = process.env.COMFYUI_NODE_MAP || "{}";
    return nodeMap;
  }

  getSamplerNameNodeId() {
    let samplerNameNodeId = this.nodeMap?.samplerName ||'';

    if(!samplerNameNodeId){
      const samplerNodeId   = this.getSamplerNodeId();
      samplerNameNodeId = samplerNodeId;
      if(this.workflow?.[samplerNodeId]?.class_type === "SamplerCustom") {
        samplerNameNodeId = this.workflow?.[samplerNodeId]?.inputs?.sampler[0];
      }
    }

    if(!this.checkProperty(this.workflow?.[samplerNameNodeId]?.inputs, "sampler_name")) {
      logger.info("sampler name node doesnt have sampler_name input.");
      samplerNameNodeId = '';
    }

    return samplerNameNodeId;
  }

  getSeedNodeId() {
    let seedNodeId = this.nodeMap?.seed || '';
    if(!seedNodeId){
      seedNodeId = this.getSamplerNodeId();
    }

    if(!(this.checkProperty(this.workflow?.[seedNodeId]?.inputs, "seed") || this.checkProperty(this.workflow?.[seedNodeId]?.inputs, "noise_seed"))) {
      logger.info("seed node doesnt have seed or noise_seed input.");
      seedNodeId = '';
    }

    return seedNodeId;
  }

  getServerURL() {
    const url = process.env.COMFYUI_URL || '';
    if (!url && !this.override) {
      throw new Error('Missing COMFYUI_URL environment variable.');
    }
    return url;
  }

  getAPIKey() {
    const api_key = process.env.COMFYUI_API_KEY || '';
    return api_key;
  }

  getWorkflow() {
    if(this.override) {
      return ''
    }

    const workflow = process.env.COMFYUI_WORKFLOW || '';
    if (!workflow) {
      throw new Error('Missing COMFYUI_WORKFLOW environment variable.');
    }
    return workflow;
  }

  getSamplerNodeId() {
    const samplerTypes  = ["SamplerCustom", "KSampler", "KSamplerAdvanced"];
    const samplerNodeId = this.nodeMap?.sampler || Object.keys(this.workflow).find(k => samplerTypes.includes(this.workflow?.[k]?.class_type)) || '';

    if(!samplerNodeId) {
      logger.info("sampler node not found.");
    }

    return samplerNodeId;
  }

  getModelNodeId() {
    let modelNodeId = this.nodeMap?.model || '';
    if(!modelNodeId){
      const samplerNodeId = this.getSamplerNodeId();
      modelNodeId         = this.workflow?.[samplerNodeId]?.inputs?.model[0];
    }

    if(!this.checkProperty(this.workflow?.[modelNodeId]?.inputs, "ckpt_name")){
      logger.info("model node doesnt have ckpt_name input.");
      modelNodeId = '';
    }

    return modelNodeId;
  }

  getPromptNodeId(name) {
    let promptNodeId = this.nodeMap?.[name] || '';
    if(!promptNodeId) {
      const samplerNodeId = this.getSamplerNodeId();
      promptNodeId        = this.workflow?.[samplerNodeId]?.inputs?.[name]?.[0] || '';
    }

    if(!this.checkProperty(this.workflow?.[promptNodeId]?.inputs, "text")) {
      logger.info(`${name} prompt node doesnt have text input.`);
      promptNodeId = '';
    }

    return promptNodeId;
  }

  getPositivePromptNodeId() {
    if(this.override){
      return '';
    }

    const nodeId = this.getPromptNodeId("positive");
    if(!nodeId){
      throw new Error("Couldn't find valid positive prompt node id");
    }
    return nodeId;
  }

  getNegativePromptNodeId() {
    const nodeId = this.getPromptNodeId("negative");

    return nodeId;
  }
}

class ComfyUiInfo extends ComfyUiBase {
  constructor(fields) {
    super(fields);

    this.name = "comfyui_info";

    this.description = 'You can get the available image generation models for comfyui with \'comfyui_info\'. Any listed model can be use for the \'image_model\' parameter for the \'comfyui_image_gen\' tool'
  }

  async getModels() {
    const url = this.url;

    let response;
    try {
      response = await axios.get(`${url}/object_info`);
    } catch (error) {
      logger.error('[ComfyUI] Error while getting object_info:', error);
      return 'Error making API request.';
    }

    const class_type = this.workflow[this.modelNodeId].class_type;
    const models     = response.data[class_type].input.required.ckpt_name[0] || [];
    return {models};
  }

  async _call() {
    const models = await this.getModels();
    return JSON.stringify(models);
  }
}

class ComfyUiAPI extends ComfyUiBase {
  constructor(fields) {
    super(fields);

    this.name = "comfyui_image_gen"

    this.description = 'You can generate images using text with \'stable-diffusion\'. This tool is exclusively for visual content.';
    this.schema = z.object({
      prompt: z
        .string()
        .describe(
          'Detailed keywords to describe the subject, using at least 7 keywords to accurately describe the image, separated by comma',
        ),
      negative_prompt: z
        .string()
        .describe(
          'Keywords we want to exclude from the final image, using at least 7 keywords to accurately describe the image, separated by comma',
        ),
      image_model: z
        .string()
        .optional()
        .describe(
          'Image generation model to use. MUST be any model listed by \'comfyui_info\'. Generally, only use this parameter if the user requests it.',
        ),
      seed: z
        .bigint()
        .optional()
        .describe(
          'Seed for image generation. Specifying the same seed will generate the same image, given the same prompt. Useful for trying varying prompts without completely changing the image. Usually, you shouldnt specify a seed explicitly',
        ),
      steps: z
        .bigint()
        .optional()
        .describe(
          'Controls the number of generation steps to run. More steps can result in higher quality images. Usually, you shouldnt specify steps explicitly. Should be 40 at max.',
        ),
    });
  }

  async _call(data) {
    const url = this.url;
    const {prompt, negative_prompt} = data;

  }

  replaceNewLinesWithSpaces(inputString) {
    return inputString.replace(/\r\n|\r|\n/g, ' ');
  }

  getMarkdownImageUrl(imageName) {
    const imageUrl = path
      .join(this.relativePath, this.userId, imageName)
      .replace(/\\/g, '/')
      .replace('public/', '');
    return `![generated image](/${imageUrl})`;
  }

  returnValue(value) {
    if (this.isAgent === true && typeof value === 'string') {
      return [value, {}];
    } else if (this.isAgent === true && typeof value === 'object') {
      return [displayMessage, value];
    }

    return value;
  }


  async _call(data) {
    const url = this.url;
    const { prompt, negative_prompt } = data;
    const payload = {
      prompt,
      negative_prompt,
      cfg_scale: 4.5,
      steps: 22,
      width: 1024,
      height: 1024,
    };
    let generationResponse;
    try {
      generationResponse = await axios.post(`${url}/sdapi/v1/txt2img`, payload);
    } catch (error) {
      logger.error('[StableDiffusion] Error while generating image:', error);
      return 'Error making API request.';
    }
    const image = generationResponse.data.images[0];

    /** @type {{ height: number, width: number, seed: number, infotexts: string[] }} */
    let info = {};
    try {
      info = JSON.parse(generationResponse.data.info);
    } catch (error) {
      logger.error('[StableDiffusion] Error while getting image metadata:', error);
    }

    const file_id = uuidv4();
    const imageName = `${file_id}.png`;
    const { imageOutput: imageOutputPath, clientPath } = paths;
    const filepath = path.join(imageOutputPath, this.userId, imageName);
    this.relativePath = path.relative(clientPath, imageOutputPath);

    if (!fs.existsSync(path.join(imageOutputPath, this.userId))) {
      fs.mkdirSync(path.join(imageOutputPath, this.userId), { recursive: true });
    }

    try {
      if (this.isAgent) {
        const content = [
          {
            type: ContentTypes.IMAGE_URL,
            image_url: {
              url: `data:image/png;base64,${image}`,
            },
          },
        ];

        const response = [
          {
            type: ContentTypes.TEXT,
            text: displayMessage,
          },
        ];
        return [response, { content }];
      }

      const buffer = Buffer.from(image.split(',', 1)[0], 'base64');
      if (this.returnMetadata && this.uploadImageBuffer && this.req) {
        const file = await this.uploadImageBuffer({
          req: this.req,
          context: FileContext.image_generation,
          resize: false,
          metadata: {
            buffer,
            height: info.height,
            width: info.width,
            bytes: Buffer.byteLength(buffer),
            filename: imageName,
            type: 'image/png',
            file_id,
          },
        });

        const generationInfo = info.infotexts[0].split('\n').pop();
        return {
          ...file,
          prompt,
          metadata: {
            negative_prompt,
            seed: info.seed,
            info: generationInfo,
          },
        };
      }

      await sharp(buffer)
        .withMetadata({
          iptcpng: {
            parameters: info.infotexts[0],
          },
        })
        .toFile(filepath);
      this.result = this.getMarkdownImageUrl(imageName);
    } catch (error) {
      logger.error('[StableDiffusion] Error while saving the image:', error);
    }

    return this.returnValue(this.result);
  }
}

function createComfyUiTools(fields = {}){
  const comfyUiBaseTool  = new ComfyUiBase(fields);
  // const comfyUiImageTool = new ComfyUiAPI(fields);
  // const comfyUiInfoTool  = new ComfyUiInfo(fields);

  // return [comfyUiInfoTool, comfyUiImageTool];
  return [comfyUiBaseTool];
}

module.exports = createComfyUiTools;
