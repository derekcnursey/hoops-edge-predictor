import { NcaaOddsData, NcaaOddsRoundKey, formatRoundOdds } from "./ncaaOdds";

const HARD_ROCK_NCAAM_PAGE_URL = "https://www.hardrock.bet/sportsbook/basketball/ncaam/";
const SNAPSHOT_LABEL = "Manual Hard Rock snapshot · March 19, 2026";

export type HardRockMarketKey =
  | "sweet-16"
  | "elite-8"
  | "final-four"
  | "champion";

type HardRockMarketInput = {
  team: string;
  odds: string;
  group: string;
};

type HardRockMarketDefinition = {
  key: HardRockMarketKey;
  label: string;
  modelRoundKey: NcaaOddsRoundKey;
  groupTarget: number;
  scope: "national" | "region";
  note: string;
  inputs: HardRockMarketInput[];
};

export type HardRockMarketRow = {
  marketKey: HardRockMarketKey;
  marketLabel: string;
  group: string;
  team: string;
  hrbTeamName: string;
  seed: number;
  region: string | null;
  hrbOdds: string;
  hrbProb: number;
  hrbFairProb: number;
  modelProb: number;
  modelOdds: string | null;
  deltaPctPoints: number;
};

export type HardRockMarketReport = {
  key: HardRockMarketKey;
  label: string;
  modelRoundKey: NcaaOddsRoundKey;
  scope: "national" | "region";
  source: "manual_snapshot";
  snapshotLabel: string;
  note: string;
  groupTarget: number;
  groupHoldPct: Record<string, number>;
  rows: HardRockMarketRow[];
  topOverlays: HardRockMarketRow[];
  topUnderlays: HardRockMarketRow[];
  topFavorites: HardRockMarketRow[];
  unmatchedTeams: string[];
  matchedCount: number;
};

export type HardRockComparisonRow = HardRockMarketRow;
export type HardRockRegionWinnerRow = HardRockMarketRow;
export type HardRockRegionWinnerReport = HardRockMarketReport;

export type HardRockComparisonData = {
  fetchedAt: string;
  sourceUrl: string;
  sportsbookPageUrl: string;
  status: "manual_snapshot";
  note: string;
  rows: HardRockComparisonRow[];
  topOverlays: HardRockComparisonRow[];
  topUnderlays: HardRockComparisonRow[];
  topByHardRock: HardRockComparisonRow[];
  unmatchedTeams: string[];
  matchedCount: number;
  regionWinnerReport: HardRockMarketReport | null;
  elite8Report: HardRockMarketReport | null;
  sweet16Report: HardRockMarketReport | null;
  championshipReport: HardRockMarketReport | null;
  reports: HardRockMarketReport[];
};

const TEAM_ALIASES: Record<string, string> = {
  "connecticut huskies": "UConn",
  "penn quakers": "Pennsylvania",
  "hawaii rainbow warriors": "Hawai'i",
  "liu sharks": "Long Island University",
  "li u sharks": "Long Island University",
  "miami redhawks": "Miami (OH)",
  "miami ohio redhawks": "Miami (OH)",
  "mcneese state cowboys": "McNeese",
  "saint marys gaels": "Saint Mary's",
  "saint johns red storm": "St. John's",
  "st johns red storm": "St. John's",
  "saint louis billikens": "Saint Louis",
  "southern methodist university mustangs": "SMU",
  "virginia commonwealth rams": "VCU",
  "brigham young cougars": "BYU",
  "miami florida hurricanes": "Miami",
  "louisville cardinals": "Louisville",
  "texas a m aggies": "Texas A&M",
  "prairie view a m panthers d1": "Prairie View A&M",
  "howard bison d1": "Howard",
};

const EAST_REGION_WINNER: HardRockMarketInput[] = [
  { group: "East", team: "Duke Blue Devils", odds: "-160" },
  { group: "East", team: "Connecticut Huskies", odds: "+500" },
  { group: "East", team: "Michigan State Spartans", odds: "+700" },
  { group: "East", team: "Saint John's Red Storm", odds: "+900" },
  { group: "East", team: "Kansas Jayhawks", odds: "+1200" },
  { group: "East", team: "Louisville Cardinals", odds: "+1400" },
  { group: "East", team: "UCLA Bruins", odds: "+2250" },
  { group: "East", team: "Ohio State Buckeyes", odds: "+3000" },
  { group: "East", team: "TCU Horned Frogs", odds: "+10000" },
  { group: "East", team: "South Florida Bulls", odds: "+15000" },
  { group: "East", team: "UCF Knights", odds: "+20000" },
  { group: "East", team: "Northern Iowa Panthers", odds: "+20000" },
  { group: "East", team: "California Baptist Lancers", odds: "+50000" },
  { group: "East", team: "North Dakota State Bison", odds: "+50000" },
  { group: "East", team: "Furman Paladins", odds: "+50000" },
  { group: "East", team: "Siena Saints", odds: "+50000" },
];

const WEST_REGION_WINNER: HardRockMarketInput[] = [
  { group: "West", team: "Arizona Wildcats", odds: "-140" },
  { group: "West", team: "Purdue Boilermakers", odds: "+375" },
  { group: "West", team: "Gonzaga Bulldogs", odds: "+500" },
  { group: "West", team: "Arkansas Razorbacks", odds: "+1000" },
  { group: "West", team: "Wisconsin Badgers", odds: "+1500" },
  { group: "West", team: "Brigham Young Cougars", odds: "+3000" },
  { group: "West", team: "Miami Florida Hurricanes", odds: "+5000" },
  { group: "West", team: "Villanova Wildcats", odds: "+6000" },
  { group: "West", team: "Missouri Tigers", odds: "+7500" },
  { group: "West", team: "Texas Longhorns", odds: "+7500" },
  { group: "West", team: "Utah State Aggies", odds: "+7500" },
  { group: "West", team: "High Point Panthers", odds: "+25000" },
  { group: "West", team: "Hawaii Rainbow Warriors", odds: "+25000" },
  { group: "West", team: "Kennesaw State Owls", odds: "+25000" },
  { group: "West", team: "Queens University Royals", odds: "+25000" },
  { group: "West", team: "LIU Sharks", odds: "+25000" },
];

const MIDWEST_REGION_WINNER: HardRockMarketInput[] = [
  { group: "Midwest", team: "Michigan Wolverines", odds: "-150" },
  { group: "Midwest", team: "Iowa State Cyclones", odds: "+300" },
  { group: "Midwest", team: "Virginia Cavaliers", odds: "+800" },
  { group: "Midwest", team: "Tennessee Volunteers", odds: "+1400" },
  { group: "Midwest", team: "Texas Tech Red Raiders", odds: "+1750" },
  { group: "Midwest", team: "Alabama Crimson Tide", odds: "+2250" },
  { group: "Midwest", team: "Kentucky Wildcats", odds: "+4000" },
  { group: "Midwest", team: "Georgia Bulldogs", odds: "+6000" },
  { group: "Midwest", team: "Santa Clara Broncos", odds: "+7500" },
  { group: "Midwest", team: "Saint Louis Billikens", odds: "+10000" },
  { group: "Midwest", team: "Miami Ohio Redhawks", odds: "+25000" },
  { group: "Midwest", team: "Akron Zips", odds: "+25000" },
  { group: "Midwest", team: "Hofstra Pride", odds: "+25000" },
  { group: "Midwest", team: "Wright State Raiders", odds: "+25000" },
  { group: "Midwest", team: "Tennessee State Tigers", odds: "+25000" },
  { group: "Midwest", team: "Howard Bison (D1)", odds: "+25000" },
];

const SOUTH_REGION_WINNER: HardRockMarketInput[] = [
  { group: "South", team: "Florida Gators", odds: "+125" },
  { group: "South", team: "Houston Cougars", odds: "+200" },
  { group: "South", team: "Illinois Fighting Illini", odds: "+325" },
  { group: "South", team: "Vanderbilt Commodores", odds: "+1200" },
  { group: "South", team: "Nebraska Cornhuskers", odds: "+1300" },
  { group: "South", team: "Saint Mary's Gaels", odds: "+4000" },
  { group: "South", team: "North Carolina Tar Heels", odds: "+5000" },
  { group: "South", team: "Iowa Hawkeyes", odds: "+5000" },
  { group: "South", team: "Clemson Tigers", odds: "+7500" },
  { group: "South", team: "Texas A&M Aggies", odds: "+15000" },
  { group: "South", team: "Virginia Commonwealth Rams", odds: "+15000" },
  { group: "South", team: "McNeese State Cowboys", odds: "+25000" },
  { group: "South", team: "Troy Trojans", odds: "+25000" },
  { group: "South", team: "Pennsylvania Quakers", odds: "+25000" },
  { group: "South", team: "Idaho Vandals", odds: "+25000" },
  { group: "South", team: "Prairie View A&M Panthers (D1)", odds: "+50000" },
];

const CHAMPIONSHIP_MARKET: HardRockMarketInput[] = [
  { group: "National", team: "Duke Blue Devils", odds: "+325" },
  { group: "National", team: "Michigan Wolverines", odds: "+350" },
  { group: "National", team: "Arizona Wildcats", odds: "+375" },
  { group: "National", team: "Florida Gators", odds: "+700" },
  { group: "National", team: "Houston Cougars", odds: "+1000" },
  { group: "National", team: "Iowa State Cyclones", odds: "+1750" },
  { group: "National", team: "Illinois Fighting Illini", odds: "+2250" },
  { group: "National", team: "Purdue Boilermakers", odds: "+2500" },
  { group: "National", team: "Connecticut Huskies", odds: "+2500" },
  { group: "National", team: "Arkansas Razorbacks", odds: "+5000" },
  { group: "National", team: "Michigan State Spartans", odds: "+5000" },
  { group: "National", team: "Gonzaga Bulldogs", odds: "+5000" },
  { group: "National", team: "Saint John's Red Storm", odds: "+6000" },
  { group: "National", team: "Kansas Jayhawks", odds: "+6000" },
  { group: "National", team: "Virginia Cavaliers", odds: "+6000" },
  { group: "National", team: "Vanderbilt Commodores", odds: "+7500" },
  { group: "National", team: "Wisconsin Badgers", odds: "+7500" },
  { group: "National", team: "Nebraska Cornhuskers", odds: "+10000" },
  { group: "National", team: "Kentucky Wildcats", odds: "+10000" },
  { group: "National", team: "Texas Tech Red Raiders", odds: "+10000" },
  { group: "National", team: "UCLA Bruins", odds: "+10000" },
  { group: "National", team: "Iowa Hawkeyes", odds: "+15000" },
  { group: "National", team: "Alabama Crimson Tide", odds: "+15000" },
  { group: "National", team: "Tennessee Volunteers", odds: "+15000" },
  { group: "National", team: "Brigham Young Cougars", odds: "+15000" },
  { group: "National", team: "Louisville Cardinals", odds: "+15000" },
  { group: "National", team: "Utah State Aggies", odds: "+20000" },
  { group: "National", team: "North Carolina Tar Heels", odds: "+20000" },
  { group: "National", team: "Miami Florida Hurricanes", odds: "+20000" },
  { group: "National", team: "Ohio State Buckeyes", odds: "+20000" },
  { group: "National", team: "Clemson Tigers", odds: "+20000" },
  { group: "National", team: "Saint Mary's Gaels", odds: "+20000" },
  { group: "National", team: "Villanova Wildcats", odds: "+20000" },
  { group: "National", team: "Texas Longhorns", odds: "+25000" },
  { group: "National", team: "Texas A&M Aggies", odds: "+25000" },
  { group: "National", team: "Missouri Tigers", odds: "+25000" },
  { group: "National", team: "Georgia Bulldogs", odds: "+25000" },
  { group: "National", team: "UCF Knights", odds: "+25000" },
  { group: "National", team: "Santa Clara Broncos", odds: "+50000" },
  { group: "National", team: "Virginia Commonwealth Rams", odds: "+50000" },
  { group: "National", team: "Kennesaw State Owls", odds: "+50000" },
  { group: "National", team: "Akron Zips", odds: "+50000" },
  { group: "National", team: "Saint Louis Billikens", odds: "+50000" },
  { group: "National", team: "South Florida Bulls", odds: "+50000" },
  { group: "National", team: "TCU Horned Frogs", odds: "+50000" },
  { group: "National", team: "Hofstra Pride", odds: "+100000" },
  { group: "National", team: "North Dakota State Bison", odds: "+100000" },
  { group: "National", team: "Queens University Royals", odds: "+100000" },
  { group: "National", team: "Siena Saints", odds: "+100000" },
  { group: "National", team: "Howard Bison (D1)", odds: "+100000" },
  { group: "National", team: "California Baptist Lancers", odds: "+100000" },
  { group: "National", team: "Idaho Vandals", odds: "+100000" },
  { group: "National", team: "McNeese State Cowboys", odds: "+100000" },
  { group: "National", team: "High Point Panthers", odds: "+100000" },
  { group: "National", team: "Troy Trojans", odds: "+100000" },
  { group: "National", team: "Northern Iowa Panthers", odds: "+100000" },
  { group: "National", team: "Pennsylvania Quakers", odds: "+100000" },
  { group: "National", team: "Wright State Raiders", odds: "+100000" },
  { group: "National", team: "Miami Ohio Redhawks", odds: "+100000" },
  { group: "National", team: "Furman Paladins", odds: "+100000" },
  { group: "National", team: "Prairie View A&M Panthers (D1)", odds: "+250000" },
  { group: "National", team: "Hawaii Rainbow Warriors", odds: "+250000" },
  { group: "National", team: "LIU Sharks", odds: "+250000" },
  { group: "National", team: "Tennessee State Tigers", odds: "+250000" },
];

const EAST_ELITE_8: HardRockMarketInput[] = [
  { group: "East", team: "Duke Blue Devils", odds: "-300" },
  { group: "East", team: "Connecticut Huskies", odds: "+150" },
  { group: "East", team: "Michigan State Spartans", odds: "+200" },
  { group: "East", team: "Louisville Cardinals", odds: "+450" },
  { group: "East", team: "Kansas Jayhawks", odds: "+500" },
  { group: "East", team: "Saint John's Red Storm", odds: "+500" },
  { group: "East", team: "UCLA Bruins", odds: "+600" },
  { group: "East", team: "Ohio State Buckeyes", odds: "+1600" },
  { group: "East", team: "South Florida Bulls", odds: "+2000" },
  { group: "East", team: "UCF Knights", odds: "+3000" },
  { group: "East", team: "TCU Horned Frogs", odds: "+3000" },
  { group: "East", team: "Northern Iowa Panthers", odds: "+10000" },
  { group: "East", team: "California Baptist Lancers", odds: "+15000" },
  { group: "East", team: "North Dakota State Bison", odds: "+15000" },
  { group: "East", team: "Furman Paladins", odds: "+15000" },
  { group: "East", team: "Siena Saints", odds: "+15000" },
];

const WEST_ELITE_8: HardRockMarketInput[] = [
  { group: "West", team: "Arizona Wildcats", odds: "-300" },
  { group: "West", team: "Purdue Boilermakers", odds: "+100" },
  { group: "West", team: "Gonzaga Bulldogs", odds: "+200" },
  { group: "West", team: "Arkansas Razorbacks", odds: "+450" },
  { group: "West", team: "Wisconsin Badgers", odds: "+650" },
  { group: "West", team: "Brigham Young Cougars", odds: "+800" },
  { group: "West", team: "Miami Florida Hurricanes", odds: "+1200" },
  { group: "West", team: "Missouri Tigers", odds: "+1750" },
  { group: "West", team: "Texas Longhorns", odds: "+2000" },
  { group: "West", team: "Utah State Aggies", odds: "+3000" },
  { group: "West", team: "Villanova Wildcats", odds: "+4000" },
  { group: "West", team: "High Point Panthers", odds: "+25000" },
  { group: "West", team: "Hawaii Rainbow Warriors", odds: "+25000" },
  { group: "West", team: "Kennesaw State Owls", odds: "+25000" },
  { group: "West", team: "Queens University Royals", odds: "+25000" },
  { group: "West", team: "LIU Sharks", odds: "+25000" },
];

const MIDWEST_ELITE_8: HardRockMarketInput[] = [
  { group: "Midwest", team: "Michigan Wolverines", odds: "-375" },
  { group: "Midwest", team: "Iowa State Cyclones", odds: "-125" },
  { group: "Midwest", team: "Virginia Cavaliers", odds: "+250" },
  { group: "Midwest", team: "Tennessee Volunteers", odds: "+450" },
  { group: "Midwest", team: "Alabama Crimson Tide", odds: "+650" },
  { group: "Midwest", team: "Texas Tech Red Raiders", odds: "+800" },
  { group: "Midwest", team: "Kentucky Wildcats", odds: "+1100" },
  { group: "Midwest", team: "Georgia Bulldogs", odds: "+2000" },
  { group: "Midwest", team: "Santa Clara Broncos", odds: "+2500" },
  { group: "Midwest", team: "Saint Louis Billikens", odds: "+3000" },
  { group: "Midwest", team: "Akron Zips", odds: "+7500" },
  { group: "Midwest", team: "Miami Ohio Redhawks", odds: "+10000" },
  { group: "Midwest", team: "Hofstra Pride", odds: "+15000" },
  { group: "Midwest", team: "Wright State Raiders", odds: "+25000" },
  { group: "Midwest", team: "Tennessee State Tigers", odds: "+25000" },
  { group: "Midwest", team: "Howard Bison (D1)", odds: "+25000" },
];

const SOUTH_ELITE_8: HardRockMarketInput[] = [
  { group: "South", team: "Florida Gators", odds: "-200" },
  { group: "South", team: "Houston Cougars", odds: "-110" },
  { group: "South", team: "Illinois Fighting Illini", odds: "+125" },
  { group: "South", team: "Vanderbilt Commodores", odds: "+375" },
  { group: "South", team: "Nebraska Cornhuskers", odds: "+550" },
  { group: "South", team: "Saint Mary's Gaels", odds: "+1400" },
  { group: "South", team: "Iowa Hawkeyes", odds: "+1500" },
  { group: "South", team: "North Carolina Tar Heels", odds: "+2000" },
  { group: "South", team: "Texas A&M Aggies", odds: "+2500" },
  { group: "South", team: "Clemson Tigers", odds: "+2500" },
  { group: "South", team: "Virginia Commonwealth Rams", odds: "+3000" },
  { group: "South", team: "McNeese State Cowboys", odds: "+15000" },
  { group: "South", team: "Troy Trojans", odds: "+25000" },
  { group: "South", team: "Pennsylvania Quakers", odds: "+25000" },
  { group: "South", team: "Idaho Vandals", odds: "+25000" },
  { group: "South", team: "Prairie View A&M Panthers (D1)", odds: "+25000" },
];

const EAST_SWEET_16: HardRockMarketInput[] = [
  { group: "East", team: "Duke Blue Devils", odds: "-700" },
  { group: "East", team: "Connecticut Huskies", odds: "-200" },
  { group: "East", team: "Michigan State Spartans", odds: "-200" },
  { group: "East", team: "Saint John's Red Storm", odds: "-105" },
  { group: "East", team: "Kansas Jayhawks", odds: "+100" },
  { group: "East", team: "Louisville Cardinals", odds: "+175" },
  { group: "East", team: "UCLA Bruins", odds: "+175" },
  { group: "East", team: "South Florida Bulls", odds: "+750" },
  { group: "East", team: "Ohio State Buckeyes", odds: "+800" },
  { group: "East", team: "UCF Knights", odds: "+900" },
  { group: "East", team: "Northern Iowa Panthers", odds: "+1500" },
  { group: "East", team: "TCU Horned Frogs", odds: "+1500" },
  { group: "East", team: "California Baptist Lancers", odds: "+3000" },
  { group: "East", team: "North Dakota State Bison", odds: "+6000" },
  { group: "East", team: "Furman Paladins", odds: "+15000" },
  { group: "East", team: "Siena Saints", odds: "+15000" },
];

const WEST_SWEET_16: HardRockMarketInput[] = [
  { group: "West", team: "Arizona Wildcats", odds: "-1000" },
  { group: "West", team: "Purdue Boilermakers", odds: "-325" },
  { group: "West", team: "Gonzaga Bulldogs", odds: "-220" },
  { group: "West", team: "Arkansas Razorbacks", odds: "-165" },
  { group: "West", team: "Wisconsin Badgers", odds: "+140" },
  { group: "West", team: "Brigham Young Cougars", odds: "+225" },
  { group: "West", team: "Miami Florida Hurricanes", odds: "+400" },
  { group: "West", team: "Texas Longhorns", odds: "+600" },
  { group: "West", team: "Missouri Tigers", odds: "+750" },
  { group: "West", team: "Utah State Aggies", odds: "+1100" },
  { group: "West", team: "Villanova Wildcats", odds: "+1300" },
  { group: "West", team: "High Point Panthers", odds: "+3000" },
  { group: "West", team: "Hawaii Rainbow Warriors", odds: "+4000" },
  { group: "West", team: "Kennesaw State Owls", odds: "+10000" },
  { group: "West", team: "Queens University Royals", odds: "+15000" },
  { group: "West", team: "LIU Sharks", odds: "+20000" },
];

const MIDWEST_SWEET_16: HardRockMarketInput[] = [
  { group: "Midwest", team: "Michigan Wolverines", odds: "-1000" },
  { group: "Midwest", team: "Iowa State Cyclones", odds: "-375" },
  { group: "Midwest", team: "Virginia Cavaliers", odds: "-165" },
  { group: "Midwest", team: "Alabama Crimson Tide", odds: "-125" },
  { group: "Midwest", team: "Texas Tech Red Raiders", odds: "+120" },
  { group: "Midwest", team: "Tennessee Volunteers", odds: "+120" },
  { group: "Midwest", team: "Kentucky Wildcats", odds: "+400" },
  { group: "Midwest", team: "Santa Clara Broncos", odds: "+900" },
  { group: "Midwest", team: "Georgia Bulldogs", odds: "+900" },
  { group: "Midwest", team: "Akron Zips", odds: "+1200" },
  { group: "Midwest", team: "Saint Louis Billikens", odds: "+1600" },
  { group: "Midwest", team: "Miami Ohio Redhawks", odds: "+2000" },
  { group: "Midwest", team: "Hofstra Pride", odds: "+3000" },
  { group: "Midwest", team: "Wright State Raiders", odds: "+10000" },
  { group: "Midwest", team: "Tennessee State Tigers", odds: "+20000" },
  { group: "Midwest", team: "Howard Bison (D1)", odds: "+20000" },
];

const SOUTH_SWEET_16: HardRockMarketInput[] = [
  { group: "South", team: "Florida Gators", odds: "-500" },
  { group: "South", team: "Illinois Fighting Illini", odds: "-400" },
  { group: "South", team: "Houston Cougars", odds: "-375" },
  { group: "South", team: "Vanderbilt Commodores", odds: "-150" },
  { group: "South", team: "Nebraska Cornhuskers", odds: "+100" },
  { group: "South", team: "Saint Mary's Gaels", odds: "+400" },
  { group: "South", team: "North Carolina Tar Heels", odds: "+550" },
  { group: "South", team: "Iowa Hawkeyes", odds: "+550" },
  { group: "South", team: "Virginia Commonwealth Rams", odds: "+800" },
  { group: "South", team: "Texas A&M Aggies", odds: "+850" },
  { group: "South", team: "Clemson Tigers", odds: "+900" },
  { group: "South", team: "McNeese State Cowboys", odds: "+1750" },
  { group: "South", team: "Troy Trojans", odds: "+4000" },
  { group: "South", team: "Idaho Vandals", odds: "+10000" },
  { group: "South", team: "Pennsylvania Quakers", odds: "+15000" },
  { group: "South", team: "Prairie View A&M Panthers (D1)", odds: "+25000" },
];

const MARKET_DEFINITIONS: HardRockMarketDefinition[] = [
  {
    key: "sweet-16",
    label: "Sweet 16",
    modelRoundKey: "sweet-16",
    groupTarget: 4,
    scope: "region",
    note:
      "Sweet 16 prices were entered manually from your March 19 Hard Rock board. Fair probability removes vig within each region and normalizes to four Sweet 16 spots per region.",
    inputs: [
      ...EAST_SWEET_16,
      ...WEST_SWEET_16,
      ...MIDWEST_SWEET_16,
      ...SOUTH_SWEET_16,
    ],
  },
  {
    key: "elite-8",
    label: "Elite 8",
    modelRoundKey: "elite-8",
    groupTarget: 2,
    scope: "region",
    note:
      "Elite 8 prices were entered manually from your March 19 Hard Rock board. Fair probability removes vig within each region and normalizes to two Elite 8 spots per region.",
    inputs: [
      ...EAST_ELITE_8,
      ...WEST_ELITE_8,
      ...MIDWEST_ELITE_8,
      ...SOUTH_ELITE_8,
    ],
  },
  {
    key: "final-four",
    label: "Region Winner",
    modelRoundKey: "final-four",
    groupTarget: 1,
    scope: "region",
    note:
      "Region winner prices were entered manually from your March 19 Hard Rock board. Fair probability removes vig within each region and compares against Hoops Edge Final Four probability.",
    inputs: [
      ...EAST_REGION_WINNER,
      ...WEST_REGION_WINNER,
      ...MIDWEST_REGION_WINNER,
      ...SOUTH_REGION_WINNER,
    ],
  },
  {
    key: "champion",
    label: "National Champion",
    modelRoundKey: "champion",
    groupTarget: 1,
    scope: "national",
    note:
      "Championship prices were entered manually from your March 19 Hard Rock board. Fair probability removes vig across the full national title market.",
    inputs: CHAMPIONSHIP_MARKET,
  },
];

function normalizeTeamName(value: string): string {
  return value
    .toLowerCase()
    .replace(/&amp;/g, "&")
    .replace(/[’']/g, "")
    .replace(/saint/g, "st")
    .replace(/[^a-z0-9]+/g, " ")
    .trim();
}

function americanToProb(odds: string): number | null {
  const parsed = Number(odds.replace("+", ""));
  if (!Number.isFinite(parsed) || parsed === 0) return null;
  return parsed > 0 ? 100 / (parsed + 100) : -parsed / (-parsed + 100);
}

function matchTeam(team: string, ncaaData: NcaaOddsData) {
  const normalized = normalizeTeamName(team);
  const aliased = TEAM_ALIASES[normalized];
  if (aliased) {
    return ncaaData.rows.find((row) => row.team === aliased) ?? null;
  }

  const exact = ncaaData.rows.find(
    (row) => normalizeTeamName(row.team) === normalized,
  );
  if (exact) return exact;

  const partialMatches = ncaaData.rows
    .map((row) => ({ row, normalized: normalizeTeamName(row.team) }))
    .filter(
      ({ normalized: local }) =>
        normalized.startsWith(`${local} `) ||
        normalized.endsWith(` ${local}`) ||
        normalized.includes(` ${local} `),
    )
    .sort((a, b) => b.normalized.length - a.normalized.length);

  return partialMatches[0]?.row ?? null;
}

function buildMarketReport(
  ncaaData: NcaaOddsData,
  definition: HardRockMarketDefinition,
): HardRockMarketReport {
  const rows: HardRockMarketRow[] = [];
  const unmatchedTeams: string[] = [];
  const groupTotals = new Map<string, number>();

  for (const input of definition.inputs) {
    const matchedTeam = matchTeam(input.team, ncaaData);
    const hrbProb = americanToProb(input.odds);
    if (!matchedTeam || hrbProb == null) {
      unmatchedTeams.push(input.team);
      continue;
    }
    groupTotals.set(input.group, (groupTotals.get(input.group) ?? 0) + hrbProb);
    rows.push({
      marketKey: definition.key,
      marketLabel: definition.label,
      group: input.group,
      team: matchedTeam.team,
      hrbTeamName: input.team,
      seed: matchedTeam.seed,
      region: matchedTeam.region ?? null,
      hrbOdds: input.odds,
      hrbProb,
      hrbFairProb: 0,
      modelProb: matchedTeam.roundProbabilities[definition.modelRoundKey],
      modelOdds: formatRoundOdds(matchedTeam.roundProbabilities[definition.modelRoundKey]),
      deltaPctPoints: 0,
    });
  }

  const finalizedRows = rows
    .map((row) => {
      const totalProb = groupTotals.get(row.group) ?? 0;
      const fairProb =
        totalProb > 0 ? (row.hrbProb / totalProb) * definition.groupTarget : row.hrbProb;
      return {
        ...row,
        hrbFairProb: fairProb,
        deltaPctPoints: (row.modelProb - fairProb) * 100,
      };
    })
    .sort((a, b) => {
      const groupDiff = a.group.localeCompare(b.group);
      if (groupDiff !== 0) return groupDiff;
      const fairDiff = b.hrbFairProb - a.hrbFairProb;
      if (fairDiff !== 0) return fairDiff;
      return a.seed - b.seed || a.team.localeCompare(b.team);
    });

  const groupHoldPct = Object.fromEntries(
    Array.from(groupTotals.entries()).map(([group, totalProb]) => [
      group,
      (totalProb - definition.groupTarget) * 100,
    ]),
  );

  return {
    key: definition.key,
    label: definition.label,
    modelRoundKey: definition.modelRoundKey,
    scope: definition.scope,
    source: "manual_snapshot",
    snapshotLabel: SNAPSHOT_LABEL,
    note: definition.note,
    groupTarget: definition.groupTarget,
    groupHoldPct,
    rows: finalizedRows,
    topOverlays: [...finalizedRows]
      .sort((a, b) => b.deltaPctPoints - a.deltaPctPoints)
      .slice(0, 10),
    topUnderlays: [...finalizedRows]
      .sort((a, b) => a.deltaPctPoints - b.deltaPctPoints)
      .slice(0, 10),
    topFavorites: [...finalizedRows]
      .sort((a, b) => b.hrbFairProb - a.hrbFairProb)
      .slice(0, 12),
    unmatchedTeams,
    matchedCount: finalizedRows.length,
  };
}

export async function fetchHardRockComparisonData(
  ncaaData: NcaaOddsData | null,
): Promise<HardRockComparisonData | null> {
  if (!ncaaData) return null;

  const reports = MARKET_DEFINITIONS.map((definition) =>
    buildMarketReport(ncaaData, definition),
  );
  const championshipReport =
    reports.find((report) => report.key === "champion") ?? null;
  const regionWinnerReport =
    reports.find((report) => report.key === "final-four") ?? null;
  const elite8Report = reports.find((report) => report.key === "elite-8") ?? null;
  const sweet16Report =
    reports.find((report) => report.key === "sweet-16") ?? null;

  return {
    fetchedAt: new Date().toISOString(),
    sourceUrl: HARD_ROCK_NCAAM_PAGE_URL,
    sportsbookPageUrl: HARD_ROCK_NCAAM_PAGE_URL,
    status: "manual_snapshot",
    note:
      "Hard Rock March markets were loaded from the March 19 manual snapshot you supplied. Fair probabilities below are no-vig normalized within each market.",
    rows: championshipReport?.rows ?? [],
    topOverlays: championshipReport?.topOverlays ?? [],
    topUnderlays: championshipReport?.topUnderlays ?? [],
    topByHardRock: championshipReport?.topFavorites ?? [],
    unmatchedTeams: Array.from(
      new Set(reports.flatMap((report) => report.unmatchedTeams)),
    ),
    matchedCount: championshipReport?.matchedCount ?? 0,
    regionWinnerReport,
    elite8Report,
    sweet16Report,
    championshipReport,
    reports,
  };
}
