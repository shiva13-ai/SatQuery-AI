import { Switch, Route } from "wouter";
import Home from "@/pages/Home";
import ComponentShowcase from "@/pages/ComponentShowcase";
import NotFound from "@/pages/NotFound";
import { ThemeProvider } from "@/contexts/ThemeContext";

function Router() {
  return (
    <Switch>
      <Route path="/" component={Home} />
      <Route path="/components" component={ComponentShowcase} />
      <Route component={NotFound} />
    </Switch>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <Router />
    </ThemeProvider>
  );
}