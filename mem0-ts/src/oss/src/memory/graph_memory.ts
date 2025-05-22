import { BM25 } from "../utils/bm25";
import { MemoryConfig } from "../types";
import { EmbedderFactory, LLMFactory } from "../utils/factory";
import { Embedder } from "../embeddings/base";
import { LLM } from "../llms/base";
import {
  DELETE_MEMORY_TOOL_GRAPH,
  EXTRACT_ENTITIES_TOOL,
  RELATIONS_TOOL,
} from "../graphs/tools";
import { EXTRACT_RELATIONS_PROMPT, getDeleteMessages } from "../graphs/utils";
import { logger } from "../utils/logger";
import { Database, Connection, QueryResult } from "kuzu";

interface SearchOutput {
  source: string;
  source_id: string;
  relationship: string;
  relation_id: string;
  destination: string;
  destination_id: string;
  similarity: number;
}

interface ToolCall {
  name: string;
  arguments: string;
}

interface LLMResponse {
  toolCalls?: ToolCall[];
}

interface Tool {
  type: string;
  function: {
    name: string;
    description: string;
    parameters: Record<string, any>;
  };
}

interface GraphMemoryResult {
  deleted_entities: any[];
  added_entities: any[];
  relations?: any[];
}

interface QueryRow {
  source: string;
  source_id: string;
  relationship: string;
  relation_id: string;
  destination: string;
  destination_id: string;
  similarity: number;
  target?: string;
  element_id?: string;
}

export class MemoryGraph {
  private config: MemoryConfig;
  private db: Database;
  private connection: Connection;
  private embeddingModel: Embedder;
  private llm: LLM;
  private structuredLlm: LLM;
  private llmProvider: string;
  private threshold: number;
  private isInitialized = false;

  constructor(config: MemoryConfig) {
    this.config = config;
    if (!config.graphStore?.config?.url) {
      throw new Error("Database configuration is incomplete");
    }

    this.db = new Database(config.graphStore.config.url);
    this.connection = new Connection(this.db);

    this.embeddingModel = EmbedderFactory.create(
      this.config.embedder.provider,
      this.config.embedder.config
    );

    this.llmProvider = "openai";
    if (this.config.llm?.provider) {
      this.llmProvider = this.config.llm.provider;
    }
    if (this.config.graphStore?.llm?.provider) {
      this.llmProvider = this.config.graphStore.llm.provider;
    }

    this.llm = LLMFactory.create(this.llmProvider, this.config.llm.config);
    this.structuredLlm = LLMFactory.create(
      "openai_structured",
      this.config.llm.config
    );
    this.threshold = 0.7;
  }

  private async init() {
    if (!this.isInitialized) {
      // Install and load vector extension
      await this.connection.query("INSTALL vector");
      await this.connection.query("LOAD vector");

      // Create vector index for each node type
      // Note: This is idempotent - if index exists, it will be skipped
      await this.connection.query(`
        CALL CREATE_VECTOR_INDEX(
          '*',                   // Index on all node types
          'idx_embedding',       // Index name
          'embedding',           // Column name containing embeddings
          metric := 'cosine',    // Using cosine similarity
          dimension := 1536      // Embedding dimension
        );
      `);

      this.isInitialized = true;
    }
  }

  async add(
    data: string,
    filters: Record<string, any>
  ): Promise<GraphMemoryResult> {
    const entityTypeMap = await this._retrieveNodesFromData(data, filters);

    const toBeAdded = await this._establishNodesRelationsFromData(
      data,
      filters,
      entityTypeMap
    );

    const searchOutput = await this._searchGraphDb(
      Object.keys(entityTypeMap),
      filters
    );

    const toBeDeleted = await this._getDeleteEntitiesFromSearchOutput(
      searchOutput,
      data,
      filters
    );

    const deletedEntities = await this._deleteEntities(
      toBeDeleted,
      filters["userId"]
    );

    const addedEntities = await this._addEntities(
      toBeAdded,
      filters["userId"],
      entityTypeMap
    );

    return {
      deleted_entities: deletedEntities,
      added_entities: addedEntities,
      relations: toBeAdded,
    };
  }

  async search(query: string, filters: Record<string, any>, limit = 100) {
    const entityTypeMap = await this._retrieveNodesFromData(query, filters);
    const searchOutput = await this._searchGraphDb(
      Object.keys(entityTypeMap),
      filters
    );

    if (!searchOutput.length) {
      return [];
    }

    const searchOutputsSequence = searchOutput.map((item) => [
      item.source,
      item.relationship,
      item.destination,
    ]);

    const bm25 = new BM25(searchOutputsSequence);
    const tokenizedQuery = query.split(" ");
    const rerankedResults = bm25.search(tokenizedQuery).slice(0, 5);

    const searchResults = rerankedResults.map((item) => ({
      source: item[0],
      relationship: item[1],
      destination: item[2],
    }));

    logger.info(`Returned ${searchResults.length} search results`);
    return searchResults;
  }

  async deleteAll(filters: Record<string, any>) {
    const query = `MATCH (n) WHERE n.user_id = '${filters["userId"]}' DETACH DELETE n`;
    await this.connection.query(query);
  }

  async getAll(filters: Record<string, any>, limit = 100) {
    const query = `
      MATCH (n)-[r]->(m)
      WHERE n.user_id = '${filters["userId"]}' AND m.user_id = '${filters["userId"]}'
      RETURN n.name AS source, type(r) AS relationship, m.name AS target
      LIMIT ${Math.floor(Number(limit))}
    `;

    const result = (await this.connection.query(query)) as QueryResult;
    const rows = (await result.getAll()) as QueryRow[];

    const finalResults = rows.map((row) => ({
      source: row.source,
      relationship: row.relationship,
      target: row.target,
    }));

    logger.info(`Retrieved ${finalResults.length} relationships`);
    return finalResults;
  }

  private async _retrieveNodesFromData(
    data: string,
    filters: Record<string, any>
  ) {
    const tools = [EXTRACT_ENTITIES_TOOL] as Tool[];
    const searchResults = await this.structuredLlm.generateResponse(
      [
        {
          role: "system",
          content: `You are a smart assistant who understands entities and their types in a given text. If user message contains self reference such as 'I', 'me', 'my' etc. then use ${filters["userId"]} as the source entity. Extract all the entities from the text. ***DO NOT*** answer the question itself if the given text is a question.`,
        },
        { role: "user", content: data },
      ],
      { type: "json_object" },
      tools
    );

    let entityTypeMap: Record<string, string> = {};
    try {
      if (typeof searchResults !== "string" && searchResults.toolCalls) {
        for (const call of searchResults.toolCalls) {
          if (call.name === "extract_entities") {
            const args = JSON.parse(call.arguments);
            for (const item of args.entities) {
              entityTypeMap[item.entity] = item.entity_type;
            }
          }
        }
      }
    } catch (e) {
      logger.error(`Error in search tool: ${e}`);
    }

    entityTypeMap = Object.fromEntries(
      Object.entries(entityTypeMap).map(([k, v]) => [
        k.toLowerCase().replace(/ /g, "_"),
        v.toLowerCase().replace(/ /g, "_"),
      ])
    );

    logger.debug(`Entity type map: ${JSON.stringify(entityTypeMap)}`);
    return entityTypeMap;
  }

  private async _establishNodesRelationsFromData(
    data: string,
    filters: Record<string, any>,
    entityTypeMap: Record<string, string>
  ) {
    let messages;
    if (this.config.graphStore?.customPrompt) {
      messages = [
        {
          role: "system",
          content:
            EXTRACT_RELATIONS_PROMPT.replace(
              "USER_ID",
              filters["userId"]
            ).replace(
              "CUSTOM_PROMPT",
              `4. ${this.config.graphStore.customPrompt}`
            ) + "\nPlease provide your response in JSON format.",
        },
        { role: "user", content: data },
      ];
    } else {
      messages = [
        {
          role: "system",
          content:
            EXTRACT_RELATIONS_PROMPT.replace("USER_ID", filters["userId"]) +
            "\nPlease provide your response in JSON format.",
        },
        {
          role: "user",
          content: `List of entities: ${Object.keys(entityTypeMap)}. \n\nText: ${data}`,
        },
      ];
    }

    const tools = [RELATIONS_TOOL] as Tool[];
    const extractedEntities = await this.structuredLlm.generateResponse(
      messages,
      { type: "json_object" },
      tools
    );

    let entities: any[] = [];
    if (typeof extractedEntities !== "string" && extractedEntities.toolCalls) {
      const toolCall = extractedEntities.toolCalls[0];
      if (toolCall && toolCall.arguments) {
        const args = JSON.parse(toolCall.arguments);
        entities = args.entities || [];
      }
    }

    entities = this._removeSpacesFromEntities(entities);
    logger.debug(`Extracted entities: ${JSON.stringify(entities)}`);
    return entities;
  }

  private async _searchGraphDb(
    nodeList: string[],
    filters: Record<string, any>,
    limit = 100
  ): Promise<SearchOutput[]> {
    const resultRelations: SearchOutput[] = [];

    for (const node of nodeList) {
      const nEmbedding = await this.embeddingModel.embed(node);

      const cypher = `
        CALL QUERY_VECTOR_INDEX(
          '*',
          'idx_embedding',
          $embedding,
          ${Math.floor(Number(limit))}
        )
        WITH node AS n, distance as similarity
        WHERE n.user_id = $user_id AND similarity >= $threshold
        MATCH (n)-[r]->(m)
        RETURN n.name AS source, elementId(n) AS source_id, 
               type(r) AS relationship, elementId(r) AS relation_id,
               m.name AS destination, elementId(m) AS destination_id,
               similarity
        ORDER BY similarity
      `;

      const rows = await this.executeQuery(cypher, {
        embedding: nEmbedding,
        user_id: filters["userId"],
        threshold: this.threshold,
      });

      resultRelations.push(...rows);
    }

    return resultRelations;
  }

  private async _getDeleteEntitiesFromSearchOutput(
    searchOutput: SearchOutput[],
    data: string,
    filters: Record<string, any>
  ) {
    const searchOutputString = searchOutput
      .map(
        (item) =>
          `${item.source} -- ${item.relationship} -- ${item.destination}`
      )
      .join("\n");

    const [systemPrompt, userPrompt] = getDeleteMessages(
      searchOutputString,
      data,
      filters["userId"]
    );

    const tools = [DELETE_MEMORY_TOOL_GRAPH] as Tool[];
    const memoryUpdates = await this.structuredLlm.generateResponse(
      [
        { role: "system", content: systemPrompt },
        { role: "user", content: userPrompt },
      ],
      { type: "json_object" },
      tools
    );

    const toBeDeleted: any[] = [];
    if (typeof memoryUpdates !== "string" && memoryUpdates.toolCalls) {
      for (const item of memoryUpdates.toolCalls) {
        if (item.name === "delete_graph_memory") {
          toBeDeleted.push(JSON.parse(item.arguments));
        }
      }
    }

    const cleanedToBeDeleted = this._removeSpacesFromEntities(toBeDeleted);
    logger.debug(
      `Deleted relationships: ${JSON.stringify(cleanedToBeDeleted)}`
    );
    return cleanedToBeDeleted;
  }

  private async _deleteEntities(toBeDeleted: any[], userId: string) {
    const results: QueryRow[][] = [];

    for (const item of toBeDeleted) {
      const { source, destination, relationship } = item;

      const cypher = `
        MATCH (n {name: $source_name, user_id: $user_id})
        -[r:${relationship}]->
        (m {name: $dest_name, user_id: $user_id})
        DELETE r
        RETURN n.name AS source, m.name AS target, type(r) AS relationship
      `;

      const rows = await this.executeQuery(cypher, {
        source_name: source,
        dest_name: destination,
        user_id: userId,
      });

      results.push(rows);
    }

    return results;
  }

  private async _addEntities(
    toBeAdded: any[],
    userId: string,
    entityTypeMap: Record<string, string>
  ) {
    const results: QueryRow[][] = [];

    for (const item of toBeAdded) {
      const { source, destination, relationship } = item;
      const sourceType = entityTypeMap[source] || "unknown";
      const destinationType = entityTypeMap[destination] || "unknown";

      const sourceEmbedding = await this.embeddingModel.embed(source);
      const destEmbedding = await this.embeddingModel.embed(destination);

      const cypher = `
        MERGE (n:${sourceType} {name: $source_name, user_id: $user_id})
        ON CREATE SET n.created = timestamp(), n.embedding = $source_embedding
        MERGE (m:${destinationType} {name: $dest_name, user_id: $user_id})
        ON CREATE SET m.created = timestamp(), m.embedding = $dest_embedding
        MERGE (n)-[r:${relationship}]->(m)
        ON CREATE SET r.created = timestamp()
        RETURN n.name AS source, type(r) AS relationship, m.name AS target
      `;

      const rows = await this.executeQuery(cypher, {
        source_name: source,
        dest_name: destination,
        source_embedding: sourceEmbedding,
        dest_embedding: destEmbedding,
        user_id: userId,
      });

      results.push(rows);
    }

    return results;
  }

  private _removeSpacesFromEntities(entityList: any[]) {
    return entityList.map((item) => ({
      ...item,
      source: item.source.toLowerCase().replace(/ /g, "_"),
      relationship: item.relationship.toLowerCase().replace(/ /g, "_"),
      destination: item.destination.toLowerCase().replace(/ /g, "_"),
    }));
  }

  private async _searchSourceNode(
    sourceEmbedding: number[],
    userId: string,
    threshold = 0.9
  ) {
    const cypher = `
      CALL QUERY_VECTOR_INDEX(
        '*',
        'idx_embedding',
        $embedding,
        1
      )
      WITH node AS source_candidate, distance as similarity
      WHERE source_candidate.user_id = $user_id AND similarity >= $threshold
      RETURN elementId(source_candidate) as element_id
    `;

    const rows = await this.executeQuery(cypher, {
      embedding: sourceEmbedding,
      user_id: userId,
      threshold,
    });

    return rows.map((row) => ({
      elementId: row.element_id?.toString() || "",
    }));
  }

  private async _searchDestinationNode(
    destinationEmbedding: number[],
    userId: string,
    threshold = 0.9
  ) {
    const cypher = `
      CALL QUERY_VECTOR_INDEX(
        '*',
        'idx_embedding',
        $embedding,
        1
      )
      WITH node AS destination_candidate, distance as similarity
      WHERE destination_candidate.user_id = $user_id AND similarity >= $threshold
      RETURN elementId(destination_candidate) as element_id
    `;

    const rows = await this.executeQuery(cypher, {
      embedding: destinationEmbedding,
      user_id: userId,
      threshold,
    });

    return rows.map((row) => ({
      elementId: row.element_id?.toString() || "",
    }));
  }

  private async executeQuery(
    query: string,
    params?: Record<string, any>
  ): Promise<QueryRow[]> {
    await this.init();
    const stmt = await this.connection.prepare(query);
    const result = (await this.connection.execute(stmt, params)) as QueryResult;
    const rows = await result.getAll();
    return rows as QueryRow[];
  }
}
