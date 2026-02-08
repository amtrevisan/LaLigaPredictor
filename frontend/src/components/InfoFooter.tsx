import { useState, useEffect } from "react";
import {
  ChevronRight,
  Linkedin,
  Github,
  Youtube,
  Instagram,
  Zap,
  Target,
  TrendingUp,
  Users,
  Shield,
  Rocket,
} from "lucide-react";

const InfoFooter = () => {
  console.log("InfoFooter rendered");
  const [animatedStats, setAnimatedStats] = useState({
    accuracy: 0,
    predictions: 0,
    components: 0,
  });

  useEffect(() => {
    const duration = 2000;
    const steps = 60;
    const interval = duration / steps;
    const targets = { accuracy: 60, predictions: 300, components: 15 };
    let step = 0;

    const timer = setInterval(() => {
      step++;
      const progress = step / steps;
      const easeOut = 1 - Math.pow(1 - progress, 3);
      setAnimatedStats({
        accuracy: Math.round(targets.accuracy * easeOut),
        predictions: Math.round(targets.predictions * easeOut),
        components: Math.round(targets.components * easeOut),
      });
      if (step >= steps) {
        clearInterval(timer);
        setAnimatedStats(targets);
      }
    }, interval);

    return () => clearInterval(timer);
  }, []);

  const socialLinks = [
    { name: "LinkedIn", url: "https://www.linkedin.com/in/alex-morales-dev/", icon: Linkedin },
    { name: "GitHub", url: "https://github.com/AlexMoralesDev", icon: Github },
    { name: "YouTube", url: "https://www.youtube.com/@alexmoralesdev", icon: Youtube },
    { name: "Instagram", url: "https://www.instagram.com/alexmoralesdev", icon: Instagram },
    {
      name: "TikTok",
      url: "https://www.tiktok.com/@alexmoralesdev",
      icon: () => (
        <svg viewBox="0 0 24 24" fill="currentColor" className="w-5 h-5">
          <path d="M19.59 6.69a4.83 4.83 0 01-3.77-4.25V2h-3.45v13.67a2.89 2.89 0 01-5.2 1.74 2.89 2.89 0 012.31-4.64 2.93 2.93 0 01.88.13V9.4a6.84 6.84 0 00-1-.05A6.33 6.33 0 005 20.1a6.34 6.34 0 0010.86-4.43v-7a8.16 8.16 0 004.77 1.52v-3.4a4.85 4.85 0 01-1-.1z" />
        </svg>
      ),
    },
  ];

  return (
    <footer className="bg-laliga-navy text-primary-foreground mt-16">
      <div className="max-w-[1400px] mx-auto px-4 py-12">
        {/* Divider */}
        <div className="border-t border-primary-foreground/10 my-8" />

        {/* Bottom section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* Project Highlights */}
          <div>
            <h3 className="font-semibold mb-3 text-sm uppercase tracking-wider">
              Project Highlights
            </h3>
            <div className="space-y-2 text-sm text-primary-foreground/60">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-success" />
                <span>Deployed web app</span>
              </div>
              <div className="flex items-center gap-2">
                <Zap className="w-4 h-4 text-warning" />
                <span>Live game predictions</span>
              </div>
              <div className="flex items-center gap-2">
                <Users className="w-4 h-4 text-primary" />
                <span>Historical predictions</span>
              </div>
            </div>
          </div>

          {/* Connect */}
          <div className="text-left md:text-right">
            <h3 className="font-semibold mb-3 text-sm uppercase tracking-wider">
              Connect With Me
            </h3>
            <div className="flex flex-wrap justify-start md:justify-end gap-2 mb-3">
              {socialLinks.map((social) => {
                const Icon = social.icon;
                return (
                  <a
                    key={social.name}
                    href={social.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="p-2 bg-primary-foreground/10 rounded-lg border border-primary-foreground/10 transition-all duration-300 hover:scale-110 hover:bg-primary-foreground/20 hover:-translate-y-0.5"
                    aria-label={social.name}
                  >
                    <Icon />
                  </a>
                );
              })}
            </div>
            <p className="text-sm text-primary-foreground/50">Alex Morales Trevisan</p>
          </div>
        </div>

        {/* Copyright */}
        <div className="text-center mt-10 pt-6 border-t border-primary-foreground/10">
          <p className="text-xs text-primary-foreground/40">
            © 2026 La Liga Predictor • Designed for analytical and educational purposes • Last Updated: February 2026
          </p>
        </div>
      </div>
    </footer>
  );
};

export default InfoFooter;
