import { NextResponse } from 'next/server';
import { runRAG } from '../../../lib/rag.js';

// POST /api/prompt
export async function POST(request) {
  try {
    // 1. Read the incoming JSON body
    const body = await request.json();
    const question = body.question;

    // 2. Validate the input
    if (!question || typeof question !== 'string') {
      return NextResponse.json(
        { error: 'Missing or invalid "question" field.' },
        { status: 400 }
      );
    }

    // 3. Run the full RAG pipeline
    const result = await runRAG(question);

    // 4. Return the result in the assignment's required format
    return NextResponse.json(result);

  } catch (err) {
    console.error('Error in /api/prompt:', err);
    return NextResponse.json(
      { error: 'Something went wrong processing your question.', detail: err.message },
      { status: 500 }
    );
  }
}

// Optional: friendly message if someone visits with GET
export async function GET() {
  return NextResponse.json({
    message: 'Send a POST request with { "question": "your question" } to use this endpoint.',
  });
}