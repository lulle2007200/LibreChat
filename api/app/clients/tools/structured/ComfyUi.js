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
const WebSocket = require('ws');

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
    this.schedulerNodeId   = this.getSchedulerNodeId();
    this.cfgNodeId         = this.getCfgNodeId();
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

  getCfgNodeId() {
    let cfgNodeId = this.nodeMap?.cfg || '';
    if(!cfgNodeId) {
      cfgNodeId = this.getSamplerNodeId();
    }

    if(!this.checkProperty(this.workflow?.[cfgNodeId]?.inputs, "cfg")) {
      logger.info("cfg node doesnt have cfg input");
      cfgNodeId = '';
    }

    return cfgNodeId;
  }

  getSchedulerNodeId() {
    let schedulerNodeId = this.nodeMap?.scheduler || '';
    if(!schedulerNodeId) {
      schedulerNodeId = this.getSamplerNodeId()
    }

    if(!this.checkProperty(this.workflow?.[schedulerNodeId]?.inputs, "scheduler")) {
      logger.info("scheduler node doesnt have scheduler input");
      schedulerNodeId = '';
    }

    return schedulerNodeId;
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

    if (!this.checkProperty(this.workflow?.[stepsNodeId]?.inputs, "steps")){
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

    if(this.checkProperty(this.workflow?.[seedNodeId]?.inputs, "seed")) {
      this.seedKey = "seed";
    } else if (this.checkProperty(this.workflow?.[seedNodeId]?.inputs, "noise_seed")) {
      this.seedKey = "noise_seed";
    } else {
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

    this.name = "comfyui-info";

    this.description = 'You can use the \'comfyui_info\' tool to get additional information about comfyui, such as available models or samplers.'
  
    this.schema = z.object({
      info_type: z
        .string()
        .describe(
`The type of information you want to get. Can be any of the following:
- 'models': Returns a list of all available image generation models.
- 'samplers': Returns a list of all available samplers.
- 'schedulers': Returns a list of all available schedulers.`
        ),
    })
  }

  async getObjectInfo() {
    const url = this.url;

    let response;
    try {
      response = await axios.get(`${url}/object_info`);
    } catch (error) {
      logger.error('[ComfyUI] Error while getting object_info:', error);
      throw error;
    }

    return response.data || {};
  }

  getModels = async() => {
    const objectInfo = await this.getObjectInfo();

    const class_type = this.workflow[this.modelNodeId]?.class_type || "CheckpointLoaderSimple";
    const models     = objectInfo[class_type]?.input?.required?.ckpt_name?.[0] || [];

    return {models};
  }

  getSamplers = async() => {
    const objectInfo = await this.getObjectInfo();

    const class_type = this.workflow[this.samplerNameNodeId]?.class_type || "KSampler";
    const samplers   = objectInfo[class_type]?.input?.required?.sampler_name?.[0] || [];

    return {samplers};
  }

  getSchedulers = async() => {
    const objectInfo = await this.getObjectInfo();
    
    const class_type = this.workflow[this.samplerNodeId]?.class_type ||"KSampler";
    const schedulers = objectInfo[class_type]?.input?.required?.scheduler?.[0] || [];

    return {schedulers};
  }

  async _call(data) {
    const functionMap = {
      "models": this.getModels,
      "samplers": this.getSamplers,
      "schedulers": this.getSchedulers,
    };

    let res;
    try {
      res = await functionMap[data.info_type]();
      res = JSON.stringify(res);
    } catch (error) {
      res = "Error making API request.";
    }
    return res;
  }
}

class ComfyUiAPI extends ComfyUiBase {
  constructor(fields) {
    super(fields);

    // this.name = "comfyui-img"
    this.name = "comfyui-img"

    this.description = 'You can generate images using text with \'stable-diffusion-comfyui\'. This tool is exclusively for visual content.';
    this.schema = z.object({
      prompt: z
        .string()
        .describe(
          'Detailed keywords to describe the subject, using at least 7 keywords to accurately describe the image, separated by comma',
        ),

      ...(this.negativeNodeId ? {
        negativePrompt: z
          .string()
          .describe(
            'Keywords we want to exclude from the final image, using at least 7 keywords to accurately describe the image, separated by comma',
          ),
      } : {}),

      ...(this.seedNodeId ? {
        seed: z
          .number()
          .int()
          .optional()
          .describe(
            'Seed for image generation. Specifying the same seed will generate the same image, given the same prompt. Useful for trying varying prompts without completely changing the image. Usually, you shouldnt specify a seed explicitly',
          ),
      } : {}),

      ...(this.modelNodeId ? {
        model: z
          .string()
          .optional()
          .describe(
            'Image generation model to use. MUST be any model listed by \'comfyui-info\'. Generally, only use this parameter if the user requests a specific model.',
          ),
      } : {}),

      ...((this.widthNodeId && this.heightNodeId) ? {
        width: z
          .number()
          .int()
          .gte(512)
          .lte(2048)
          .optional()
          .describe(
            'Width of the image to generate. MUST be between 512 and 2048. This parameter is optional, the default is 512.',
          ),
        height: z
          .number()
          .int()
          .gte(512)
          .lte(2048)
          .optional()
          .describe(
            'height of the image to generate. MUST be between 512 and 2048. This parameter is optional, the default is 512.',
          ),
      } : {}),

      ...(this.samplerNameNodeId ? {
        sampler: z
          .string()
          .optional()
          .describe(
            'The sampler to use for image generation. MUST be any sampler listed by \'comfyui-info\'. Generally, only use this parameter if the user requests a specific sampler.',
          ),
      } : {}),

      ...(this.schedulerNodeId ? {
        scheduler: z
          .string()
          .optional()
          .describe(
            'The scheduler to use for image generation. MUST be any sampler listed by \'comfyui-info\'. Generally, only use this parameter if the user requests a specific scheduler.',
          ),
      } : {}),

      ...(this.cfgNodeId ? {
        cfg: z
          .number()
          .int()
          .gte(0)
          .lte(20)
          .optional()
          .describe(
            'Controls creative the model is/how much it adheres to the prompts. The higher, the more prompt adherance. Default is 4. This parameter is optional.',
          ),
      } : {}),

      ...(this.stepsNodeId ? {
        steps: z
          .number()
          .int()
          .gte(5)
          .lte(40)
          .optional()
          .describe(
            'Controls the number of generation steps to run. More steps can result in higher quality images. Usually, you shouldnt specify steps explicitly. Should be 40 at max.',
          ),
      } : {}),

    });
  }

  async queuePrompt(workflow, callId) {
    const url = this.url;

    let headers = {}
    if(this.api_key) {
      headers["Authorization"] = `Bearer ${this.api_key}`
    }

    let response;
    try {
      response = await axios.post(`${url}/prompt`, {client_id: callId, prompt: workflow}, {headers: headers});
      return response.data.prompt_id;
    } catch (error) {
      logger.error("[ComfyUI] Error queueing new prompt.");
      throw error;
    }
  }

  async getHistory(promptId) {
    const url = this.url;

    let headers = {}
    if(this.api_key) {
      headers["Authorization"] = `Bearer ${this.api_key}`
    }

    let response;
    try {
      response = await axios.get(`${url}/history/${promptId}`, {headers: headers});
      return response.data;
    } catch (error) {
      logger.error("[ComfyUI] Error getting prompt history.");
      throw error;
    }
  }

  getImageUrl(image) {
    const params = new URLSearchParams({
      filename: image.filename,
      subfolder: image.subfolder,
      type: image.type,
    });

    return `${this.url}/view?${params.toString()}`;
  }

  async getImages(workflow, ws, callId) {
    const promptId = await this.queuePrompt(workflow, callId);

    logger.info(`[ComfyUI] Submitted prompt ${promptId}.`);


    while(true) {
      const data = await new Promise((resolve, reject) => {
        const onMessage = (data, isBinary) => {
          cleanup();
          resolve({data, isBinary});
        }

        const onError = (err) => {
          cleanup();
          reject(err);
        }

        const cleanup = () => {
          ws.off("message", onMessage);
          ws.off("error", onError);
        }
        ws.on("message", onMessage);
        ws.on("error", onError);
      });

      if(data.isBinary) {
        continue;
      }

      const msgStr = data.data.toString();
      const msg    = JSON.parse(msgStr);

      if(msg.data.prompt_id !== promptId) {
        continue;
      }

      if(msg.type === "execution_success") {
        logger.info(`[ComfyUI] ${promptId} successfully generated.`);
        break;
      }
    }

    const history = await this.getHistory(promptId);
    const output  = history[promptId].outputs;

    let outputImages = [];

    for(const nodeKey in output) {
      const node = output[nodeKey];
      if(node.images) {
        for(const image of node.images) {
          const url = this.getImageUrl(image);

          outputImages.push(url);
        }
      }
    }

    return outputImages;
  }

  async _call(data) {
    // return JSON.parse(testImage);
    const url = this.url;
    const ws_url = url.replace("https://", "wss://").replace("http://", "ws://");

    let workflow = this.workflow;

    workflow[this.positiveNodeId].inputs.text = data.prompt;

    if(data.negativePrompt && this.negativeNodeId) {
      workflow[this.negativeNodeId].inputs.text = data.negativePrompt;
    }

    if(data.model && this.modelNodeId) {
      workflow[this.modelNodeId].inputs.ckpt_name = data.model;
    }

    if(data.sampler && this.samplerNameNodeId) {
      workflow[this.samplerNameNodeId].inputs.sampler_name = data.sampler;
    }

    if(data.steps && this.stepsNodeId) {
      workflow[this.stepsNodeId].inputs.steps = data.steps;
    }

    if(data.width && this.widthNodeId) {
      workflow[this.widthNodeId].inputs.width = data.width;
    }

    if(data.height && this.heightNodeId) {
      workflow[this.widthNodeId].inputs.height = data.height;
    }

    if(this.seedNodeId) {
      if(data.seed) {
        workflow[this.seedNodeId].inputs[this.seedKey] = data.seed;
      }else {
        workflow[this.seedNodeId].inputs[this.seedKey] = Math.floor(Math.random() * 0x100000000);
      }
    }

    if(data.scheduler && this.schedulerNodeId) {
      workflow[this.schedulerNodeId].inputs.scheduler = data.scheduler;
    }

    if(data.cfg && this.cfgNodeId) {
      workflow[this.cfgNodeId].inputs.cfg = data.cfg;
    }

    let headers = {}
    if(this.api_key) {
      headers["Authorization"] = `Bearer ${this.api_key}`
    }

    let ws;
    const callId = crypto.randomUUID();
    try {
      ws = new WebSocket(`${ws_url}/ws?clientId=${callId}`, {headers: headers});


      await new Promise((resolve, reject) => {
        if(ws.readyState === WebSocket.OPEN) {
          return resolve(ws);
        }
        if(ws.readyState === WebSocket.CLOSING || ws.readyState === WebSocket.CLOSED) {
          return reject(new Error("[ComfyUI] Websocket already closed."));
        }
        const timer = setTimeout(() => reject(new Error("[ComfyUI] Websocket timeout.")), 5000);
        ws.onopen = () => {
          clearTimeout(timer);
          resolve(ws);
        };
        ws.onerror = (err) => {
          clearTimeout(timer);
          reject(err);
        };
      });

    } catch (error) {
      logger.error("[ComfyUI] Error while opening WebSocket:", error);
      return 'Error making API request.'
    }

    let outputImages;
    try{
      outputImages = await this.getImages(workflow, ws, callId);
      ws.close();
    } catch (error) {
      logger.error("[ComfyUI] Error while opening WebSocket:", error);
      ws.close();
      return 'Error making API request.'
    }

    // const { imageOutput: imageOutputPath, clientPath } = paths;
    // if (!fs.existsSync(path.join(imageOutputPath, this.userId))) {
    //   fs.mkdirSync(path.join(imageOutputPath, this.userId), {recursive : true});
    // }

    try {
      let content = [];
      for(const outputImage of outputImages) {
        const response = await axios.get(outputImage, {headers: headers, responseType: "arraybuffer"});
        const b64Image = Buffer.from(response.data, 'binary').toString('base64');

        content.push({
          type: ContentTypes.IMAGE_URL,
          image_url: {
            url: `data:image/png;base64,${b64Image}`,
          },
        });
      }


      const displayMessage = 'ComfyUI displayed one or more images. All images are already plainly visible, so don\'t repeat the descriptions in detail. Do not list download links as they are available in the UI already. The user may download the images by clicking on them, but do not mention anything about downloading to the user.';
      const response = [
        {
          type: ContentTypes.TEXT,
          text: displayMessage,
        },
      ];

      const ret = [response, { content }];
      return [response, { content }];
    } catch (error) {
      logger.error("[ComfyUI] Error getting generated images:", error);
      return 'Error making API request.'
    }
  } 
}

function createComfyUiTools(fields = {}) {
  const comfyUiInfoTool = new ComfyUiInfo(fields);
  const comfyUiImageTool = new ComfyUiAPI(fields);

  return [comfyUiInfoTool, comfyUiImageTool];
}

module.exports = createComfyUiTools;
