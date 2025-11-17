"""
CodeSense AI - API Routes
Handles code analysis requests using Google Gemini AI
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import re
import httpx
from typing import Optional, List, Dict

# Try to use Google Generative AI SDK if available
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

router = APIRouter(prefix="/api", tags=["api"])


class CodeAnalysisRequest(BaseModel):
    code: str
    language: str


class ErrorItem(BaseModel):
    line: int
    message: str
    severity: str


class CodeAnalysisResponse(BaseModel):
    errors: List[ErrorItem]
    suggestions: List[str]
    optimizations: List[str]
    output: str


# --- Debug endpoints ---
@router.get("/debug/ping")
async def debug_ping():
    return {"ok": True}


@router.get("/debug/models")
async def debug_models():
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        return JSONResponse(status_code=500, content={"error": "GEMINI_API_KEY is not configured"})
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            for version in ["v1", "v1beta"]:
                url = f"https://generativelanguage.googleapis.com/{version}/models?key={gemini_api_key}"
                resp = await client.get(url)
                if resp.status_code == 200:
                    data = resp.json()
                    models = [m["name"] for m in data.get("models", [])]
                    return {"api_version": version, "count": len(models), "models": models[:10]}
            # If none succeeded, return the last one
            return JSONResponse(status_code=resp.status_code, content={"error": await resp.aread()})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@router.post("/analyze")
async def analyze_code(request: CodeAnalysisRequest):
    """
    Analyze code using Google Gemini AI
    
    Args:
        request: CodeAnalysisRequest containing code and language
        
    Returns:
        CodeAnalysisResponse with errors, suggestions, optimizations, and output
    """
    if not request.code or not request.language:
        raise HTTPException(
            status_code=400,
            detail="Code and language are required"
        )
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not configured"
        )
    
    # Verify API key format
    if not gemini_api_key.startswith("AIza"):
        raise HTTPException(
            status_code=500,
            detail="Invalid GEMINI_API_KEY format. API key should start with 'AIza'"
        )
    
    prompt = f"""Analyze the following {request.language} code and provide a comprehensive analysis in JSON format. 

Code to analyze:
```{request.language}
{request.code}
```

Please provide your response in this exact JSON structure:
{{
  "errors": [
    {{
      "line": number,
      "message": "description of the error",
      "severity": "error" | "warning" | "info"
    }}
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
}}

Focus on:
1. Syntax errors, logic errors, and potential runtime issues
2. Best practices and code quality improvements  
3. Performance optimizations and cleaner code suggestions
4. What the code output would be (if any print/console statements exist)

Be thorough but concise. Only include actual issues, not hypothetical ones."""

    try:
        # Skip SDK for now - use direct API calls (more reliable)
        use_sdk = False  # Temporarily disabled to avoid async issues
        generated_text = None
        
        if use_sdk and False:  # Disabled
            try:
                # Run SDK calls in executor to avoid blocking async event loop
                import asyncio
                import concurrent.futures
                
                def run_sdk_sync():
                    try:
                        genai.configure(api_key=gemini_api_key)
                        
                        # First, try to list available models to see what we can use
                        try:
                            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                            print(f"Available models: {available_models}")
                            # Filter to gemini models
                            gemini_models = [m for m in available_models if 'gemini' in m.lower()]
                            if gemini_models:
                                return gemini_models[:3]  # Try first 3 available
                            else:
                                return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                        except Exception as e:
                            print(f"Model listing failed: {e}")
                            # If listing fails, use default list
                            return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
                    except Exception as e:
                        print(f"SDK configuration failed: {e}")
                        return None
                
                # Use ThreadPoolExecutor for SDK calls
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    models_to_try = await loop.run_in_executor(executor, run_sdk_sync)
                
                if not models_to_try:
                    raise Exception("Failed to get models list")
                
                def try_model_sync(model_name):
                    # Extract just the model name if it's a full path
                    if '/' in model_name:
                        model_name = model_name.split('/')[-1]
                    print(f"Trying SDK with model: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    response_obj = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": 0.1,
                            "top_k": 32,
                            "top_p": 1,
                            "max_output_tokens": 2048,
                        }
                    )
                    return response_obj.text
                
                # Try each model
                loop = asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for model_name in models_to_try:
                        try:
                            generated_text = await loop.run_in_executor(
                                executor, try_model_sync, model_name
                            )
                            print(f"[SUCCESS] SDK connected to model: {model_name}")
                            break
                        except Exception as e:
                            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                            print(f"[FAILED] SDK with {model_name}: {error_msg}")
                            if model_name == models_to_try[-1]:
                                raise
                            continue
            except Exception as sdk_error:
                error_msg = str(sdk_error).encode('ascii', 'replace').decode('ascii')
                print(f"SDK failed, falling back to direct API: {error_msg}")
                use_sdk = False
                generated_text = None
        
        # Fallback to direct API calls
        if not use_sdk or generated_text is None:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Try the most likely working options first
                combinations_to_try = [
                    ("v1", "gemini-2.5-flash"),
                    ("v1beta", "gemini-2.5-flash"),
                    ("v1", "gemini-1.5-flash"),
                ]
                
                for api_version, model_name in combinations_to_try:
                    try:
                        api_url = f"https://generativelanguage.googleapis.com/{api_version}/models/{model_name}:generateContent?key={gemini_api_key}"
                        print(f"Trying: {api_version}/{model_name}")
                        
                        response = await client.post(
                            api_url,
                            json={
                                "contents": [{
                                    "role": "user",
                                    "parts": [{
                                        "text": prompt
                                    }]
                                }],
                                "generationConfig": {
                                    "temperature": 0.1,
                                    "topK": 32,
                                    "topP": 1,
                                    "maxOutputTokens": 2048,
                                }
                            },
                            headers={
                                "Content-Type": "application/json"
                            }
                        )
                        
                        if response.is_success:
                            print(f"[SUCCESS] Direct API worked with {api_version}/{model_name}")
                            break
                        else:
                            error_text = await response.text()
                            safe_error = error_text[:300].encode('ascii', 'replace').decode('ascii')
                            print(f"[FAILED] {api_version}/{model_name} failed ({response.status_code}): {safe_error}")
                            # Only store non-404 errors
                            if response.status_code != 404:
                                last_error = f"{api_version}/{model_name}: {safe_error}"
                            response = None
                    except Exception as e:
                        error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                        print(f"[ERROR] Exception with {api_version}/{model_name}: {error_msg}")
                        if "404" not in error_msg:
                            last_error = error_msg
                        continue
                    
                    if response and response.is_success:
                        break
                
                if not response or not response.is_success:
                    error_detail = last_error or "No available models found. Please check your API key has access to Gemini models."
                    print(f"All API attempts failed. Last error: {error_detail}")
                    return JSONResponse(status_code=(response.status_code if response else 500), content={
                        "detail": f"Gemini API error: {error_detail}",
                        "errors": [{"line": 1, "message": f"Gemini API error: {error_detail}", "severity": "error"}],
                        "suggestions": ["Verify API key and model access"],
                        "optimizations": ["None"],
                        "output": "Analysis unavailable"
                    })
                
                # Parse JSON response robustly
                try:
                    data = response.json()
                    # Extract text directly from API response if available
                    if generated_text is None:
                        try:
                            cand0 = data.get("candidates", [])[0]
                            content = cand0.get("content", {}) if isinstance(cand0, dict) else {}
                            parts = content.get("parts", []) if isinstance(content, dict) else []
                            if parts and isinstance(parts[0], dict) and isinstance(parts[0].get("text"), str):
                                generated_text = parts[0]["text"]
                        except Exception:
                            pass
                except Exception:
                    # If response is not JSON, surface first 300 chars for diagnostics
                    raw_text = await response.text()
                    safe_preview = raw_text[:300]
                    raise HTTPException(
                        status_code=500,
                        detail=f"Gemini API returned non-JSON response for {api_version}/{model_name}: {safe_preview}"
                    )
        
        # Get generated text from API response if not already set
        if generated_text is None:
            # Get from API response
            if not data or not data.get("candidates") or not data["candidates"][0].get("content"):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response from Gemini API"
                )
            content_obj = data["candidates"][0].get("content", {})
            parts = content_obj.get("parts", []) if isinstance(content_obj, dict) else []
            if parts and isinstance(parts[0], dict):
                if "text" in parts[0] and isinstance(parts[0]["text"], str):
                    generated_text = parts[0]["text"]
                else:
                    # If it's not plain text, serialize the first part
                    generated_text = json.dumps(parts[0])
            else:
                # Fallback: serialize entire content
                generated_text = json.dumps(content_obj)
        
        # Safe print for Windows console
        try:
            safe_text = generated_text[:100].encode('ascii', 'replace').decode('ascii')
            print(f"Generated text (first 100 chars): {safe_text}")
        except:
            print("Generated text received (encoding safe)")
        
        # Ensure generated_text is a string for regex and parsing
        if not isinstance(generated_text, str):
            generated_text = json.dumps(generated_text)

        # Extract JSON from the response (handle code blocks)
        try:
            json_match = (
                re.search(r'```json\n([\s\S]*?)\n```', generated_text) or
                re.search(r'```\n([\s\S]*?)\n```', generated_text) or
                None
            )
            
            if json_match:
                json_string = json_match.group(1)
            else:
                json_string = generated_text
            
            # json_string must be a string here; attempt to parse
            if isinstance(json_string, str):
                analysis_result = json.loads(json_string)
            else:
                analysis_result = {"output": str(json_string)}
        except (json.JSONDecodeError, AttributeError) as parse_error:
            safe_error = str(parse_error).encode('ascii', 'replace').decode('ascii')
            print(f"Failed to parse Gemini response as JSON: {safe_error}")
            # Fallback response structure
            analysis_result = {
                "errors": [],
                "suggestions": ["AI analysis failed to parse. Please check your code syntax."],
                "optimizations": ["Consider reviewing your code structure."],
                "output": "Analysis unavailable"
            }
            
            # Ensure the response has the expected structure
            errors_list = []
            if isinstance(analysis_result.get("errors"), list):
                for error in analysis_result.get("errors", []):
                    if isinstance(error, dict):
                        # Ensure all required fields are present
                        errors_list.append(ErrorItem(
                            line=error.get("line", 1),
                            message=error.get("message", "Unknown error"),
                            severity=error.get("severity", "error")
                        ))
            
            result = {
                "errors": errors_list,
                "suggestions": (
                    analysis_result.get("suggestions", ["No suggestions available"])
                    if isinstance(analysis_result.get("suggestions"), list)
                    else ["No suggestions available"]
                ),
                "optimizations": (
                    analysis_result.get("optimizations", ["No optimizations suggested"])
                    if isinstance(analysis_result.get("optimizations"), list)
                    else ["No optimizations suggested"]
                ),
                "output": analysis_result.get("output", "No output detected")
            }
            
            print(f"Final analysis result: {result}")
            return JSONResponse(status_code=200, content=result)
            
    except httpx.TimeoutException:
        error_msg = "Request to Gemini API timed out"
        print(f"[ERROR] {error_msg}")
        return JSONResponse(status_code=504, content={
            "detail": error_msg,
            "errors": [{"line": 1, "message": error_msg, "severity": "error"}],
            "suggestions": ["Retry in a moment"],
            "optimizations": ["None"],
            "output": "Analysis unavailable"
        })
    except HTTPException as he:
        # Return HTTPException details as JSON payload for the frontend
        print(f"[ERROR] HTTPException: {he.detail}")
        return JSONResponse(status_code=he.status_code, content={
            "detail": str(he.detail),
            "errors": [{"line": 1, "message": str(he.detail), "severity": "error"}],
            "suggestions": ["Check API key/model access"],
            "optimizations": ["None"],
            "output": "Analysis error"
        })
    except Exception as error:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(error).encode('ascii', 'replace').decode('ascii')
        print(f"[ERROR] Error in analyze-code function:")
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] Traceback:\n{error_trace}")
        return JSONResponse(status_code=500, content={
            "detail": f"Internal server error: {error_msg}",
            "errors": [{"line": 1, "message": f"Internal server error: {error_msg}", "severity": "error"}],
            "suggestions": ["See server logs for details"],
            "optimizations": ["None"],
            "output": "Analysis error"
        })

    # Final safety net: never return None
    print("[WARN] analyze_code reached end without explicit return; sending generic error JSON")
    return JSONResponse(status_code=500, content={
        "detail": "Unknown error: no response generated",
        "errors": [{"line": 1, "message": "Unknown error: no response generated", "severity": "error"}],
        "suggestions": ["Retry", "Check server logs"],
        "optimizations": ["None"],
        "output": "Analysis unavailable"
    })

