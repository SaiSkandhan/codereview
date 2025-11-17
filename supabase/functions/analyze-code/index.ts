import "https://deno.land/x/xhr@0.1.0/mod.ts";
import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
};

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { code, language } = await req.json();
    
    if (!code || !language) {
      return new Response(
        JSON.stringify({ error: 'Code and language are required' }),
        { 
          status: 400, 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        }
      );
    }

    const GEMINI_API_KEY = Deno.env.get('GEMINI_API_KEY');
    if (!GEMINI_API_KEY) {
      throw new Error('GEMINI_API_KEY is not configured');
    }

    const prompt = `Analyze the following ${language} code and provide a comprehensive analysis in JSON format. 

Code to analyze:
\`\`\`${language}
${code}
\`\`\`

Please provide your response in this exact JSON structure:
{
  "errors": [
    {
      "line": number,
      "message": "description of the error",
      "severity": "error" | "warning" | "info"
    }
  ],
  "suggestions": [
    "suggestion 1",
    "suggestion 2"
  ],
  "optimizations": [
    "optimization 1",
    "optimization 2"
  ],
  "output": "expected output or 'No output detected'"
}

Focus on:
1. Syntax errors, logic errors, and potential runtime issues
2. Best practices and code quality improvements  
3. Performance optimizations and cleaner code suggestions
4. What the code output would be (if any print/console statements exist)

Be thorough but concise. Only include actual issues, not hypothetical ones.`;

    console.log('Sending request to Gemini API...');

    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${GEMINI_API_KEY}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{
            text: prompt
          }]
        }],
        generationConfig: {
          temperature: 0.1,
          topK: 32,
          topP: 1,
          maxOutputTokens: 2048,
        }
      }),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Gemini API error:', errorText);
      throw new Error(`Gemini API error: ${response.status}`);
    }

    const data = await response.json();
    console.log('Gemini API response:', data);

    if (!data.candidates || !data.candidates[0] || !data.candidates[0].content) {
      throw new Error('Invalid response from Gemini API');
    }

    const generatedText = data.candidates[0].content.parts[0].text;
    console.log('Generated text:', generatedText);

    // Extract JSON from the response (handle code blocks)
    let analysisResult;
    try {
      const jsonMatch = generatedText.match(/```json\n([\s\S]*?)\n```/) || 
                       generatedText.match(/```\n([\s\S]*?)\n```/) ||
                       [null, generatedText];
      
      const jsonString = jsonMatch[1] || generatedText;
      analysisResult = JSON.parse(jsonString);
    } catch (parseError) {
      console.error('Failed to parse Gemini response as JSON:', parseError);
      // Fallback response structure
      analysisResult = {
        errors: [],
        suggestions: ["AI analysis failed to parse. Please check your code syntax."],
        optimizations: ["Consider reviewing your code structure."],
        output: "Analysis unavailable"
      };
    }

    // Ensure the response has the expected structure
    const result = {
      errors: Array.isArray(analysisResult.errors) ? analysisResult.errors : [],
      suggestions: Array.isArray(analysisResult.suggestions) ? analysisResult.suggestions : ["No suggestions available"],
      optimizations: Array.isArray(analysisResult.optimizations) ? analysisResult.optimizations : ["No optimizations suggested"],
      output: analysisResult.output || "No output detected"
    };

    console.log('Final analysis result:', result);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in analyze-code function:', error);
    return new Response(
      JSON.stringify({ 
        error: error.message,
        errors: [],
        suggestions: ["Analysis failed. Please try again."],
        optimizations: ["Unable to analyze code at this time."],
        output: "Analysis error"
      }),
      {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      }
    );
  }
});