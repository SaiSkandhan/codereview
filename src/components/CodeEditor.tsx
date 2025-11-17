import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Play, Zap, AlertCircle, CheckCircle, Lightbulb } from "lucide-react";
import { supabase } from "@/integrations/supabase/client";
import { useToast } from "@/components/ui/use-toast";

const CodeEditor = () => {
  const [selectedLanguage, setSelectedLanguage] = useState<string>("");
  const [code, setCode] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<any>(null);
  const { toast } = useToast();

  const languages = [
    { value: "javascript", label: "JavaScript" },
    { value: "python", label: "Python" },
    { value: "java", label: "Java" },
    { value: "cpp", label: "C++" },
    { value: "csharp", label: "C#" },
    { value: "typescript", label: "TypeScript" },
    { value: "go", label: "Go" },
    { value: "rust", label: "Rust" },
  ];

  const handleAnalyze = async () => {
    if (!code.trim() || !selectedLanguage) return;
    
    setIsAnalyzing(true);
    
    try {
      const { data, error } = await supabase.functions.invoke('analyze-code', {
        body: {
          code: code.trim(),
          language: selectedLanguage
        }
      });

      if (error) {
        console.error('Supabase function error:', error);
        throw new Error(error.message || 'Failed to analyze code');
      }

      console.log('Analysis result:', data);
      setAnalysisResult(data);
      
      toast({
        title: "Analysis Complete",
        description: "Your code has been analyzed by Gemini AI",
      });

    } catch (error) {
      console.error('Error analyzing code:', error);
      toast({
        title: "Analysis Failed",  
        description: "Failed to analyze code. Please try again.",
        variant: "destructive",
      });
      
      // Fallback to show error state
      setAnalysisResult({
        errors: [{ line: 1, message: "Failed to analyze code", severity: "error" }],
        suggestions: ["Please try again or check your internet connection"],
        optimizations: ["Analysis unavailable"],
        output: "Analysis failed"
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Editor Panel */}
        <div className="lg:col-span-2">
          <Card className="p-6">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-semibold">Code Editor</h2>
              <div className="flex items-center gap-3">
                <Select value={selectedLanguage} onValueChange={setSelectedLanguage}>
                  <SelectTrigger className="w-40">
                    <SelectValue placeholder="Select language" />
                  </SelectTrigger>
                  <SelectContent>
                    {languages.map((lang) => (
                      <SelectItem key={lang.value} value={lang.value}>
                        {lang.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <Button 
                  onClick={handleAnalyze} 
                  disabled={!code.trim() || !selectedLanguage || isAnalyzing}
                  className="bg-gradient-primary"
                >
                  {isAnalyzing ? (
                    <Zap className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Play className="w-4 h-4 mr-2" />
                  )}
                  {isAnalyzing ? "Analyzing..." : "Analyze Code"}
                </Button>
              </div>
            </div>
            
            <div className="relative">
              <Textarea
                value={code}
                onChange={(e) => setCode(e.target.value)}
                placeholder="Paste your code here..."
                className="min-h-[400px] font-mono text-sm bg-editor-bg border-editor-border resize-none"
              />
              <div className="absolute top-2 left-2 text-xs text-muted-foreground">
                {code.split('\n').map((_, index) => (
                  <div key={index} className="leading-6">
                    {index + 1}
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </div>

        {/* Analysis Panel */}
        <div className="space-y-6">
          {/* Code Output */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Play className="w-5 h-5 text-success" />
              Code Output
            </h3>
            {analysisResult?.output ? (
              <div className="bg-editor-bg border border-editor-border rounded-lg p-4">
                <pre className="text-sm font-mono whitespace-pre-wrap">
                  {analysisResult.output}
                </pre>
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Output will appear here after code analysis.</p>
            )}
          </Card>

          {/* Errors & Warnings */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <AlertCircle className="w-5 h-5 text-destructive" />
              Issues Found
              {analysisResult?.errors && analysisResult.errors.length > 0 && (
                <Badge variant="destructive" className="ml-2">
                  {analysisResult.errors.length}
                </Badge>
              )}
            </h3>
            {analysisResult?.errors && analysisResult.errors.length > 0 ? (
              <div className="space-y-3">
                {analysisResult.errors.map((error: any, index: number) => (
                  <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-muted/50 border-l-4 border-l-destructive">
                    <Badge 
                      variant={
                        error.severity === "error" ? "destructive" : 
                        error.severity === "warning" ? "secondary" : 
                        "outline"
                      }
                    >
                      Line {error.line}
                    </Badge>
                    <div className="flex-1">
                      <p className="text-sm font-medium">{error.message}</p>
                      <p className="text-xs text-muted-foreground capitalize">{error.severity}</p>
                    </div>
                  </div>
                ))}
              </div>
            ) : analysisResult ? (
              <div className="flex items-center gap-2 text-success">
                <CheckCircle className="w-4 h-4" />
                <p className="text-sm">No issues found! Your code looks good.</p>
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">No issues found yet. Analyze your code to see results.</p>
            )}
          </Card>

          {/* Suggestions */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-warning" />
              Suggestions
            </h3>
            {analysisResult?.suggestions ? (
              <div className="space-y-3">
                {analysisResult.suggestions.map((suggestion: string, index: number) => (
                  <div key={index} className="p-3 rounded-lg bg-warning/10 border border-warning/20">
                    <p className="text-sm">{suggestion}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Suggestions will appear here after analysis.</p>
            )}
          </Card>

          {/* Optimizations */}
          <Card className="p-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <CheckCircle className="w-5 h-5 text-success" />
              Optimizations
            </h3>
            {analysisResult?.optimizations ? (
              <div className="space-y-3">
                {analysisResult.optimizations.map((optimization: string, index: number) => (
                  <div key={index} className="p-3 rounded-lg bg-success/10 border border-success/20">
                    <p className="text-sm">{optimization}</p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-muted-foreground text-sm">Optimization tips will appear here after analysis.</p>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
};

export default CodeEditor;