export interface DemoUser {
  id: string;
  name: string;
  persona: string;
  avatar: string;
  badgeColor: string;
  description: string;
  seedPreferences: string[];
}

export const DEMO_USERS: DemoUser[] = [
  {
    id: "AE3RQLFSVY5DOCCDWJIQRQVCDS4Q",
    name: "Alex Rivera",
    persona: "Hardcore PC Gamer & Streamer",
    avatar: "🎮",
    badgeColor: "bg-indigo-500/20 text-indigo-300 border-indigo-500/30",
    description: "Frequent buyer of mechanical keyboards, RGB peripherals, gaming headsets, and GPU accessories.",
    seedPreferences: ["Video Games", "Mechanical Keyboards", "Headsets"]
  },
  {
    id: "AE3XM7WK4XUB6EDVNL2ZWTNL24EA",
    name: "Elena Rostova",
    persona: "Audiophile & Music Producer",
    avatar: "🎧",
    badgeColor: "bg-emerald-500/20 text-emerald-300 border-emerald-500/30",
    description: "Focuses on studio monitor headphones, DACs, lossless audio gear, and acoustic accessories.",
    seedPreferences: ["Electronics", "Studio Audio", "Cables"]
  },
  {
    id: "AE4SX2IMZGVQSS5SFWP6NEE2HQ4Q",
    name: "Marcus Chen",
    persona: "Smart Home & IoT Enthusiast",
    avatar: "💡",
    badgeColor: "bg-amber-500/20 text-amber-300 border-amber-500/30",
    description: "Automates everything with smart plugs, Zigbee sensors, mesh Wi-Fi routers, and smart lighting.",
    seedPreferences: ["Smart Home", "Networking", "Lighting"]
  },
  {
    id: "AE4ZGC4ABSLLPZDBLUE4B22JMGKA",
    name: "Sarah Jenkins",
    persona: "Remote Work & Ergonomics Pro",
    avatar: "💻",
    badgeColor: "bg-cyan-500/20 text-cyan-300 border-cyan-500/30",
    description: "Equipping home offices with ergonomic mice, monitor arms, USB-C docks, and desk mats.",
    seedPreferences: ["Office Products", "Computer Accessories", "Docks"]
  },
  {
    id: "AE5VPKX64CVOPH6X75C7EQXAIGLA",
    name: "Devon Vance",
    persona: "Outdoor Tech & Action Cam Explorer",
    avatar: "📷",
    badgeColor: "bg-rose-500/20 text-rose-300 border-rose-500/30",
    description: "Gears up with rugged power banks, action camera mounts, waterproof cases, and portable GPS.",
    seedPreferences: ["Camera & Photo", "Power Accessories", "Outdoor"]
  },
  {
    id: "AECTQQX663PTF5UQ2RA5TUL3BXVQ",
    name: "Aisha Patel",
    persona: "Mobile Tech & Fast Charging Power User",
    avatar: "⚡",
    badgeColor: "bg-purple-500/20 text-purple-300 border-purple-500/30",
    description: "Always testing GaN multi-port chargers, braided 240W USB-C cables, and MagSafe power banks.",
    seedPreferences: ["Cell Phones & Accessories", "Fast Charging", "MagSafe"]
  },
  {
    id: "guest_cold_start",
    name: "Guest Visitor (Cold Start)",
    persona: "New User / Zero History",
    avatar: "👤",
    badgeColor: "bg-slate-500/20 text-slate-300 border-slate-500/30",
    description: "No historical interactions — system activates cold-start heuristics & popularity baselines.",
    seedPreferences: ["Trending Items", "Top Rated"]
  }
];

export const DEFAULT_USER = DEMO_USERS[0];
