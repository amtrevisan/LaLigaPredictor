import { ChevronRight } from "lucide-react";

const HeroSection = () => {
  console.log("HeroSection rendered");
  return (
    <section className="laliga-hero-gradient py-10 md:py-16 relative overflow-hidden">
      {/* Background decorative elements */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-0 right-0 w-96 h-96 bg-warning rounded-full -translate-y-1/2 translate-x-1/2" />
        <div className="absolute bottom-0 left-0 w-64 h-64 bg-warning rounded-full translate-y-1/2 -translate-x-1/2" />
      </div>

      <div className="max-w-[1400px] mx-auto px-4 relative z-10">
        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-primary-foreground/70 text-sm mb-4">
          <a href="#" className="hover:text-primary-foreground transition-colors">Home</a>
          <ChevronRight size={14} />
          <span className="text-primary-foreground">Predictor</span>
        </div>

        {/* Title */}
        <h1 className="text-4xl md:text-5xl lg:text-6xl font-heading text-primary-foreground leading-tight mb-4">
          MATCH PREDICTIONS LALIGA<br />
          SEASON 2025/26
        </h1>
      </div>
    </section>
  );
};

export default HeroSection;