import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ArrowRight, Code, Zap, Shield } from "lucide-react";

const Hero = () => {
  return (
    <section className="py-20 px-4">
      <div className="container mx-auto text-center">
        <Badge variant="secondary" className="mb-6">
          🚀 AI-Powered Code Analysis
        </Badge>
        
        <h1 className="text-5xl lg:text-7xl font-bold mb-6 bg-gradient-hero bg-clip-text text-transparent">
          CodeSense AI
        </h1>
        
        <p className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto">
          Analyze, optimize, and improve your code with advanced AI. 
          Get instant feedback, error detection, and optimization suggestions 
          for multiple programming languages.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
          <Button size="lg" className="bg-gradient-primary shadow-glow">
            Start Analyzing Code
            <ArrowRight className="w-5 h-5 ml-2" />
          </Button>
          <Button variant="outline" size="lg">
            View Documentation
          </Button>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <div className="p-6 rounded-xl bg-card border border-border">
            <div className="w-12 h-12 bg-gradient-primary rounded-lg flex items-center justify-center mb-4 mx-auto">
              <Code className="w-6 h-6 text-primary-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Multi-Language Support</h3>
            <p className="text-muted-foreground text-sm">
              Support for JavaScript, Python, Java, C++, and more programming languages.
            </p>
          </div>
          
          <div className="p-6 rounded-xl bg-card border border-border">
            <div className="w-12 h-12 bg-gradient-accent rounded-lg flex items-center justify-center mb-4 mx-auto">
              <Zap className="w-6 h-6 text-accent-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Real-time Analysis</h3>
            <p className="text-muted-foreground text-sm">
              Get instant feedback on code quality, errors, and optimization opportunities.
            </p>
          </div>
          
          <div className="p-6 rounded-xl bg-card border border-border">
            <div className="w-12 h-12 bg-gradient-primary rounded-lg flex items-center justify-center mb-4 mx-auto">
              <Shield className="w-6 h-6 text-primary-foreground" />
            </div>
            <h3 className="text-lg font-semibold mb-2">Security Focused</h3>
            <p className="text-muted-foreground text-sm">
              Identify security vulnerabilities and best practices for secure coding.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;