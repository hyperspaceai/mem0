import { LLMConfig } from "../types";

export interface KuzuConfig {
  url: string | null;
}

export interface GraphStoreConfig {
  provider: string;
  config: KuzuConfig;
  llm?: LLMConfig;
  customPrompt?: string;
}

export function validateKuzuConfig(config: KuzuConfig): void {
  const { url } = config;
  if (!url) {
    throw new Error("Please provide 'url'.");
  }
}

export function validateGraphStoreConfig(config: GraphStoreConfig): void {
  const { provider } = config;
  if (provider === "kuzu") {
    validateKuzuConfig(config.config);
  } else {
    throw new Error(`Unsupported graph store provider: ${provider}`);
  }
}
