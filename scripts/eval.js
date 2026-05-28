import { createReadStream } from 'fs';
import { parse } from 'csv-parse';
import OpenAI from 'openai';
import { Pinecone } from '@pinecone-database/pinecone';
import * as dotenv from 'dotenv';

dotenv.config({ path: '.env.local' });

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
  baseURL: process.env.OPENAI_BASE_URL,
});

const pinecone = new Pinecone({ apiKey: process.env.PINECONE_API_KEY });
const index = pinecone.index(process.env.PINECONE_INDEX_NAME);

const LIMIT = 100;       // same 100 articles for every config
const TOP_K = 5;

// ─── CONFIGS TO COMPARE ────────────────────────────
// We test 3 settings spanning the allowed range
// (max chunk 1024, max overlap 0.3).
const CONFIGS = [
  { name: 'cfg_512_20',  chunkSize: 512,  overlap: 0.2  },
  { name: 'cfg_256_15',  chunkSize: 256,  overlap: 0.15 },
  { name: 'cfg_1024_30', chunkSize: 1024, overlap: 0.3  },
];
// ──────────────────────────────────────────────────

// ─── TEST QUESTIONS (the 4 assignment types) ───────
const QUESTIONS = [
  {
    type: '1. Precise fact retrieval',
    q: 'Find an article that reframes marketing as a conversation with readers, aimed at writers who find self-promotion uncomfortable.',
  },
  {
    type: '2. Multi-result topic listing',
    q: 'articles about writing and creativity',
  },
  {
    type: '3. Key idea summary',
    q: 'an article that argues showing up and writing consistently every day leads to success',
  },
  {
    type: '4. Recommendation',
    q: 'practical beginner-friendly advice on building habits that actually stick',
  },
];
// ──────────────────────────────────────────────────

function chunkText(text, chunkSize, overlapRatio) {
  const words = text.split(/\s+/).filter(Boolean);
  const overlapWords = Math.floor(chunkSize * overlapRatio);
  const chunks = [];
  let start = 0;
  while (start < words.length) {
    const end = Math.min(start + chunkSize, words.length);
    const chunk = words.slice(start, end).join(' ').trim();
    if (chunk) chunks.push(chunk);
    if (end === words.length) break;
    start += chunkSize - overlapWords;
  }
  return chunks;
}

async function embed(text) {
  const res = await openai.embeddings.create({
    model: '4UHRUIN-text-embedding-3-small',
    input: text,
  });
  return res.data[0].embedding;
}

async function loadArticles() {
  const articles = [];
  await new Promise((resolve, reject) => {
    createReadStream('./medium-english-50mb.csv')
      .pipe(parse({ columns: true, skip_empty_lines: true }))
      .on('data', (row) => {
        if (articles.length < LIMIT) articles.push(row);
      })
      .on('end', resolve)
      .on('error', reject);
  });
  return articles;
}

// Embed the 100 articles into a namespace for one config.
// Skips if the namespace is already populated (saves tokens/money).
async function ingestConfig(config, articles) {
  const ns = index.namespace(config.name);

  // Check if this namespace already has vectors
  const stats = await index.describeIndexStats();
  const existing = stats.namespaces?.[config.name]?.recordCount || 0;
  if (existing > 0) {
    console.log(`✅ ${config.name} already has ${existing} vectors — skipping embed`);
    return;
  }

  console.log(`🔪 Embedding 100 articles for ${config.name} (chunk=${config.chunkSize}, overlap=${config.overlap})...`);

  for (let i = 0; i < articles.length; i++) {
    const text = String(articles[i].text || '').trim();
    if (!text) continue;

    const chunks = chunkText(text, config.chunkSize, config.overlap);
    const vectors = [];

    for (let j = 0; j < chunks.length; j++) {
      const embedding = await embed(chunks[j]);
      vectors.push({
        id: `${config.name}-art${i}-ch${j}`,
        values: embedding,
        metadata: {
          article_id: String(i),
          title: String(articles[i].title || ''),
          chunk: chunks[j].slice(0, 200),
        },
      });
    }

    if (vectors.length > 0) {
      await ns.upsert({ records: vectors });
    }
    if ((i + 1) % 20 === 0) console.log(`   ...${i + 1}/100 articles done`);
  }

  console.log(`✅ ${config.name} embedded.`);
}

// Query one config's namespace, dedupe by article, return top distinct articles
async function queryConfig(config, queryVector) {
  const ns = index.namespace(config.name);
  const results = await ns.query({
    vector: queryVector,
    topK: TOP_K * 5,
    includeMetadata: true,
  });

  const matches = results.matches || [];
  const seen = new Map();
  for (const m of matches) {
    const id = m.metadata.article_id;
    if (!seen.has(id)) seen.set(id, m);
  }
  return Array.from(seen.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, TOP_K);
}

async function main() {
  console.log('📖 Loading articles...');
  const articles = await loadArticles();
  console.log(`✅ Loaded ${articles.length} articles\n`);

  // 1. Embed each config (skips if already done)
  for (const config of CONFIGS) {
    await ingestConfig(config, articles);
  }
  console.log('');

  // 2. Pre-embed the questions ONCE (reuse across configs)
  console.log('🔢 Embedding test questions...');
  const questionVectors = [];
  for (const item of QUESTIONS) {
    questionVectors.push(await embed(item.q));
  }
  console.log('✅ Done\n');

  // 3. Run each question against each config
  for (let qi = 0; qi < QUESTIONS.length; qi++) {
    const item = QUESTIONS[qi];
    console.log('═══════════════════════════════════════════════════');
    console.log(`❓ ${item.type}`);
    console.log(`   "${item.q}"`);
    console.log('═══════════════════════════════════════════════════');

    for (const config of CONFIGS) {
      const matches = await queryConfig(config, questionVectors[qi]);
      const top1 = matches[0]?.score || 0;
      const mean = matches.reduce((s, m) => s + m.score, 0) / (matches.length || 1);

      console.log(`\n  ▸ ${config.name}  (top1=${top1.toFixed(4)}, mean=${mean.toFixed(4)})`);
      matches.forEach((m, i) => {
        console.log(`      ${i + 1}. [${m.score.toFixed(4)}] ${m.metadata.title}`);
      });
    }
    console.log('');
  }

  console.log('🎉 Evaluation complete!');
  console.log('💡 Compare top1/mean scores AND whether the right articles appear.');
}

main().catch(console.error);