import { useEffect, useState, useRef } from "react";

import Chat from "./components/Chat.tsx";
import ArrowRightIcon from "./components/icons/ArrowRightIcon.tsx";
import StopIcon from "./components/icons/StopIcon.tsx";
import Progress from "./components/Progress.tsx";
import { Message } from "@huggingface/transformers";
import { modelNames, dtypes, devices } from "./const.ts";

// @ts-expect-error navigator.gpu is an experimental technology
const IS_WEBGPU_AVAILABLE = !!navigator.gpu;
const STICKY_SCROLL_THRESHOLD = 120;

function App() {
  // Create a reference to the worker object.
  const worker = useRef<Worker>(null);

  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Model loading and progress
  const [status, setStatus] = useState("");
  const [loadingMessage, setLoadingMessage] = useState("");
  const [progressItems, setProgressItems] = useState<
    {
      file: string;
      progress: number;
      total: number;
    }[]
  >();
  const [isRunning, setIsRunning] = useState(false);

  // Inputs and outputs
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [tps, setTps] = useState(0);
  const [numTokens, setNumTokens] = useState(0);

  // Added states
  const [modelName, setModelName] = useState<(typeof modelNames)[number]>(
    "schmuell/DeepSeek-R1-Distill-Qwen-1.5B-onnx"
  );
  const [dtype, setDtype] = useState<(typeof dtypes)[number]>("q4f16");
  const [device, setDevice] = useState<(typeof devices)[number]>("webgpu");
  const [error, setError] = useState("");

  function onEnter(message: string) {
    setMessages((prev) => prev?.concat({ role: "user", content: message }));
    setTps(0);
    setIsRunning(true);

    setInput("");
  }

  useEffect(() => {
    resizeInput();
  }, [input]);

  function onInterrupt() {
    /*
     * NOTE: We do not set isRunning to false here because the worker
     * will send a 'complete' message when it is done.
     */
    worker.current?.postMessage({ type: "interrupt" });
  }

  function resizeInput() {
    if (!textareaRef.current) return;

    const target = textareaRef.current;
    target.style.height = "auto";
    const newHeight = Math.min(Math.max(target.scrollHeight, 24), 200);
    target.style.height = `${newHeight}px`;
  }

  // We use the `useEffect` hook to setup the worker as soon as the `App` component is mounted.
  useEffect(() => {
    if (!worker.current) {
      // Create the worker if it does not yet exist.
      worker.current = new Worker(new URL("./worker.ts", import.meta.url), {
        type: "module",
      });
    }

    // Create a callback function for messages from the worker thread.
    const onMessageReceived = (e: MessageEvent) => {
      switch (e.data.status) {
        case "loading":
          // Model file start load: add a new progress item to the list.
          setStatus("loading");
          setLoadingMessage(e.data.data);
          break;

        case "initiate":
          setProgressItems((prev) => prev?.concat(e.data));
          break;

        case "progress":
          // Model file progress: update one of the progress items.
          setProgressItems((prev) =>
            prev?.map((item) => {
              if (item.file === e.data.file) {
                return { ...item, ...e.data };
              }
              return item;
            })
          );
          break;

        case "done":
          // Model file loaded: remove the progress item from the list.
          setProgressItems((prev) =>
            prev?.filter((item) => item.file !== e.data.file)
          );
          break;

        case "ready":
          // Pipeline ready: the worker is ready to accept messages.
          setStatus("ready");
          break;

        case "start":
          {
            // Start generation
            setMessages((prev) =>
              prev?.concat({ role: "assistant", content: "" })
            );
          }
          break;

        case "update":
          {
            /*
             * Generation update: update the output text.
             * Parse messages
             */
            const { output, tps, numTokens } = e.data;
            setTps(tps);
            setNumTokens(numTokens);
            setMessages((prev) => {
              if (prev === undefined) return [];
              const cloned = prev.concat([]);
              const last = cloned.at(-1);
              if (last === undefined) return [];
              cloned[cloned.length - 1] = {
                ...last,
                content: last.content + output,
              };
              return cloned;
            });
          }
          break;

        case "complete":
          // Generation complete: re-enable the "Generate" button
          setIsRunning(false);
          break;

        case "error":
          // Error from worker
          setError(e.data.error);
      }
    };

    // Attach the callback function as an event listener.
    worker.current.addEventListener("message", onMessageReceived);

    // Define a cleanup function for when the component is unmounted.
    return () => {
      worker.current?.removeEventListener("message", onMessageReceived);
    };
  }, []);

  // Send the messages to the worker thread whenever the `messages` state changes.
  useEffect(() => {
    if (messages?.filter((x) => x.role === "user").length === 0) {
      // No user messages yet: do nothing.
      return;
    }
    if (messages?.at(-1)?.role === "assistant") {
      // Do not update if the last message is from the assistant
      return;
    }
    setTps(0);
    worker.current?.postMessage({
      type: "generate",
      data: messages,
      config: { modelName, dtype, device },
    });
  }, [device, dtype, messages, modelName]);

  useEffect(() => {
    if (!chatContainerRef.current) return;
    if (isRunning) {
      const element = chatContainerRef.current;
      if (
        element.scrollHeight - element.scrollTop - element.clientHeight <
        STICKY_SCROLL_THRESHOLD
      ) {
        element.scrollTop = element.scrollHeight;
      }
    }
  }, [messages, isRunning]);

  useEffect(() => {
    document.title = `${modelName} (${dtype}) ${device}`;
  }, [device, dtype, modelName]);

  return IS_WEBGPU_AVAILABLE ? (
    <div className="flex flex-col h-screen mx-auto items justify-end text-gray-800 dark:text-gray-200 bg-white dark:bg-gray-900">
      {status == "" && messages && messages.length === 0 && (
        <div className="h-full overflow-auto scrollbar-thin flex justify-center items-center flex-col relative">
          <div className="flex flex-col items-center mb-1 max-w-[250px] text-center">
            <h1 className="text-4xl font-bold mb-1">WebNN Chat</h1>
            <h2 className="font-semibold">
              A private and powerful AI chatbot that runs locally in your
              browser.
            </h2>
          </div>

          <div className="flex flex-col items-center px-4">
            <p className="max-w-[514px] mb-4">
              Models will be cached and reused when you revisit the page.
              Everything runs directly in your browser using{" "}
              <a
                href="https://huggingface.co/docs/transformers.js"
                target="_blank"
                rel="noreferrer"
                className="underline"
              >
                🤗&nbsp;Transformers.js
              </a>{" "}
              and ONNX Runtime Web, meaning your conversations aren&#39;t sent
              to a server. You can even disconnect from the internet after the
              model has loaded!
            </p>

            <label>
              Model
              <select
                className="border px-4 py-2 mx-4 my-4 rounded-lg dark:bg-gray-900 dark:text-white"
                value={modelName}
                onChange={({ currentTarget: t }) =>
                  (modelNames as ReadonlyArray<string>).includes(t.value) &&
                  setModelName(t.value as (typeof modelNames)[number])
                }
              >
                {modelNames.map((name) => (
                  <option value={name} key={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>

            <span className="">
              <label>
                DataType
                <select
                  className="border px-4 py-2 mx-4 my-4 rounded-lg dark:bg-gray-900 dark:text-white"
                  value={dtype}
                  onChange={({ currentTarget: t }) =>
                    (dtypes as ReadonlyArray<string>).includes(t.value) &&
                    setDtype(t.value as (typeof dtypes)[number])
                  }
                >
                  {dtypes.map((name) => (
                    <option value={name} key={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>

              <label>
                Device
                <select
                  className="border px-4 py-2 mx-4 my-4 rounded-lg dark:bg-gray-900 dark:text-white"
                  value={device}
                  onChange={({ currentTarget: t }) =>
                    (devices as ReadonlyArray<string>).includes(t.value) &&
                    setDevice(t.value as (typeof devices)[number])
                  }
                >
                  {devices.map((name) => (
                    <option value={name} key={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </span>

            <button
              className="border px-4 py-2 rounded-lg bg-blue-400 text-white hover:bg-blue-500 disabled:bg-blue-100 disabled:cursor-not-allowed select-none"
              onClick={() => {
                worker.current?.postMessage({
                  type: "load",
                  config: { modelName, dtype, device },
                });
                setStatus("loading");
              }}
              disabled={status !== ""}
            >
              Load model
            </button>
          </div>
        </div>
      )}
      {status === "loading" && (
        <>
          <div className="w-full max-w-[500px] text-left mx-auto p-4 bottom-0 mt-auto">
            <p className="text-center whitespace-pre mb-1">{loadingMessage}</p>
            {progressItems?.map(({ file, progress, total }, i) => (
              <Progress
                key={i}
                text={file}
                percentage={progress}
                total={total}
              />
            ))}
          </div>
        </>
      )}

      {status === "ready" && (
        <div
          ref={chatContainerRef}
          className="overflow-y-auto scrollbar-thin w-full flex flex-col items-center h-full"
        >
          <Chat messages={messages} />
          <p className="text-center text-sm min-h-6 text-gray-500 dark:text-gray-300">
            {tps > 0 && messages.length > 0 && (
              <>
                {!isRunning && numTokens > 0 && (
                  <span>
                    Generated {numTokens} tokens in{" "}
                    {(numTokens / tps).toFixed(2)} seconds&nbsp;&#40;
                  </span>
                )}
                {
                  <>
                    <span className="font-medium text-center mr-1 text-black dark:text-white">
                      {tps.toFixed(2)}
                    </span>
                    <span className="text-gray-500 dark:text-gray-300">
                      tokens/second
                    </span>
                  </>
                }
                {!isRunning && (
                  <>
                    <span className="mr-1">&#41;.</span>
                    <span
                      className="underline cursor-pointer"
                      onClick={() => {
                        worker.current?.postMessage({ type: "reset" });
                        setMessages([]);
                      }}
                    >
                      Reset
                    </span>
                  </>
                )}
              </>
            )}
          </p>
        </div>
      )}

      <div className="mt-2 border dark:bg-gray-700 rounded-lg w-[600px] max-w-[80%] max-h-[200px] mx-auto relative mb-3 flex">
        <textarea
          ref={textareaRef}
          className="scrollbar-thin w-[550px] dark:bg-gray-700 px-3 py-4 rounded-lg bg-transparent border-none outline-none text-gray-800 disabled:text-gray-400 dark:text-gray-200 placeholder-gray-500 dark:placeholder-gray-400 disabled:placeholder-gray-200 resize-none disabled:cursor-not-allowed"
          placeholder="Type your message..."
          rows={1}
          value={input}
          disabled={status !== "ready"}
          title={status === "ready" ? "Model is ready" : "Model not loaded yet"}
          onKeyDown={(e) => {
            if (
              input.length > 0 &&
              !isRunning &&
              e.key === "Enter" &&
              !e.shiftKey
            ) {
              e.preventDefault(); // Prevent default behavior of Enter key
              onEnter(input);
            }
          }}
          onInput={(e) => setInput(e.currentTarget.value)}
        />
        {isRunning ? (
          <div className="cursor-pointer" onClick={onInterrupt}>
            <StopIcon className="h-8 w-8 p-1 rounded-md text-gray-800 dark:text-gray-100 absolute right-3 bottom-3" />
          </div>
        ) : input.length > 0 ? (
          <div className="cursor-pointer" onClick={() => onEnter(input)}>
            <ArrowRightIcon
              className={
                "h-8 w-8 p-1 bg-gray-800 dark:bg-gray-100 text-white dark:text-black rounded-md absolute right-3 bottom-3"
              }
            />
          </div>
        ) : (
          <div>
            <ArrowRightIcon
              className={
                "h-8 w-8 p-1 bg-gray-200 dark:bg-gray-600 text-gray-50 dark:text-gray-800 rounded-md absolute right-3 bottom-3"
              }
            />
          </div>
        )}
      </div>

      {error !== "" && <p className="text-red-400 text-center mb-3">{error}</p>}

      <p className="text-xs text-gray-400 text-center mb-3">
        Disclaimer: Generated content may be inaccurate or false.
      </p>
    </div>
  ) : (
    <div className="fixed w-screen h-screen bg-black z-10 bg-opacity-[92%] text-white text-2xl font-semibold flex justify-center items-center text-center">
      WebGPU is not supported
      <br />
      by this browser :&#40;
    </div>
  );
}

export default App;
