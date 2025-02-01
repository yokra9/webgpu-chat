
import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  PreTrainedTokenizer,
  ProgressCallback,
  PreTrainedModel,
  Message,
  StoppingCriteriaList,
  Tensor
} from "@huggingface/transformers";
import { modelNames, dtypes, devices } from "./const.ts";

type Config = {
  modelName: (typeof modelNames)[number];
  dtype: (typeof dtypes)[number];
  device: (typeof devices)[number];
}

class CallbackTextStreamer extends TextStreamer {
  cb: (text: string) => void;

  constructor(tokenizer: PreTrainedTokenizer, cb: (text: string) => void) {
    super(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true,
    });
    this.cb = cb;
  }

  on_finalized_text(text: string) {
    this.cb(text);
  }
}

class InterruptableStoppingCriteria extends StoppingCriteriaList {
  interrupted: boolean

  constructor() {
    super();
    this.interrupted = false;
  }

  interrupt() {
    this.interrupted = true;
  }

  reset() {
    this.interrupted = false;
  }

  _call(input_ids: number[][], _: number[][]) {
    return new Array(input_ids.length).fill(this.interrupted);
  }
}

const stopping_criteria = new InterruptableStoppingCriteria();


/**
 * This class uses the Singleton pattern to ensure that only one instance of the model is loaded.
 */
// eslint-disable-next-line @typescript-eslint/no-extraneous-class
class TextGenerationPipeline {
  static model_id: string;
  static model: Promise<PreTrainedModel>;
  static tokenizer: Promise<PreTrainedTokenizer>;
  static streamer = null;

  static async getInstance(config: Config, progress_callback?: ProgressCallback) {

    this.model_id = config.modelName;

    this.tokenizer ??= AutoTokenizer.from_pretrained(
      this.model_id,
      {
        legacy: true,
        progress_callback
      }
    );

    this.model ??= AutoModelForCausalLM.from_pretrained(
      this.model_id,
      {
        dtype: config.dtype,
        device: config.device,
        use_external_data_format: true,
        progress_callback
      }
    );

    return Promise.all([
      this.tokenizer,
      this.model
    ]);
  }
}

async function generate(messages: Message[], config: Config) {
  // Retrieve the text-generation pipeline.
  const [
    tokenizer,
    model
  ] = await TextGenerationPipeline.getInstance(config);

  const inputs = tokenizer.apply_chat_template(
    messages,
    {
      add_generation_prompt: true,
      return_dict: true
    }
  );
  if (typeof inputs !== "object") return;

  let startTime;
  let numTokens = 0;
  const cb = (output: string) => {
    startTime ??= performance.now();

    let tps;
    if (numTokens++ > 0) {
      tps = numTokens / (performance.now() - startTime) * 1000;
    }
    self.postMessage({
      status: "update",
      output,
      tps,
      numTokens
    });
  };

  const streamer = new CallbackTextStreamer(
    tokenizer,
    cb
  );
  const generation_config = model._prepare_generation_config(
    null,
    {
      max_new_tokens: 512
    }
  );

  // Tell the main thread we are starting
  self.postMessage({ status: "start" });

  const outputs = await model.generate({
    ...inputs,
    generation_config,
    streamer,
    stopping_criteria
  }) as Tensor;
  const outputText = tokenizer.batch_decode(
    outputs,
    { skip_special_tokens: false }
  );

  // Send the output back to the main thread
  self.postMessage({
    status: "complete",
    output: outputText
  });

}

async function load(config: Config) {

  self.postMessage({
    status: "loading",
    data: `Loading model...\n${config.modelName} (${config.dtype}) ${config.device}`
  });

  // Load the pipeline and save it for future use.
  const [
    tokenizer,
    model
  ] = await TextGenerationPipeline.getInstance(config, (x) => {

    /*
     * We also add a progress callback to the pipeline so that we can
     * track model loading.
     */
    self.postMessage(x);

  });

  self.postMessage({
    status: "loading",
    data: "Compiling shaders and warming up model..."
  });

  // Run model with dummy input to compile shaders
  const inputs = tokenizer("a");
  await model.generate({
    ...inputs,
    max_new_tokens: 1
  });
  self.postMessage({ "status": "ready" });
}
// Listen for messages from the main thread
self.addEventListener(
  "message",
  async (e) => {
    const { type, data, config } = e.data;
    try {
      switch (type) {
        case "load":
          load(config as Config);
          break;

        case "generate":
          stopping_criteria.reset();
          generate(data, config);
          break;

        case "interrupt":
          stopping_criteria.interrupt();
          break;

        case "reset":
          stopping_criteria.reset();
          break;
      }

    } catch (error) {
      self.postMessage({ "status": "error", error: String(error) });
    }
  }
);
