import { Button } from "@/components/ui/button";
import { Brain, Github, Star } from "lucide-react";

const Header = () => {
  return (
    <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 bg-gradient-primary rounded-lg flex items-center justify-center">
            <Brain className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-xl font-bold bg-gradient-hero bg-clip-text text-transparent">
            CodeSense AI
          </span>
        </div>
        
        <nav className="flex items-center gap-6">
          <Button variant="ghost" size="sm">
            Features
          </Button>
          <Button variant="ghost" size="sm">
            Documentation
          </Button>
          <Button variant="ghost" size="sm">
            <Github className="w-4 h-4 mr-2" />
            GitHub
          </Button>
          <Button variant="default" size="sm" className="bg-gradient-primary shadow-glow">
            <Star className="w-4 h-4 mr-2" />
            Star
          </Button>
        </nav>
      </div>
    </header>
  );
};

export default Header;