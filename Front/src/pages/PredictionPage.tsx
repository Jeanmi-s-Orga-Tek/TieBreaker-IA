import React, { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Loader2, TrendingUp, User } from "lucide-react";

interface PredictionResult {
  player1: string;
  player2: string;
  winner_prediction: string;
  probability: number;
  confidence: string;
}

export default function PredictionPage() {
  const [player1, setPlayer1] = useState("");
  const [player2, setPlayer2] = useState("");
  const [surface, setSurface] = useState("Hard");
  const [round, setRound] = useState("");
  const [tournament, setTournament] = useState("");
  const [year, setYear] = useState("");
  const [matchDate, setMatchDate] = useState("");
  const [allYears, setAllYears] = useState(false);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handlePredict = async () => {
    if (!player1 || !player2) {
      setError("Veuillez entrer les noms des deux joueurs");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const payload: Record<string, unknown> = {
        player1_name: player1,
        player2_name: player2,
        surface: surface,
      };

      if (round) payload.round = round;
      if (tournament) payload.tournament = tournament;
      if (matchDate) payload.date = matchDate;
      if (allYears) payload.all_years = true;

      if (year && !allYears) {
        const yearNumber = Number(year);
        if (!Number.isNaN(yearNumber)) {
          payload.year = yearNumber;
        }
      }

      console.log("🎾 Envoi de la requête de prédiction:", payload);

      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      console.log("📡 Réponse reçue - Status:", response.status);

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }));
        throw new Error(errorData.detail || `Erreur ${response.status}`);
      }

      const data = await response.json();
      console.log("✅ Données de prédiction reçues:", data);
      console.log("   - Vainqueur:", data.winner_prediction);
      console.log("   - Probabilité:", (data.probability * 100).toFixed(1) + "%");
      console.log("   - Confiance:", data.confidence);

      setResult(data);
    } catch (err) {
      console.error("❌ Erreur lors de la prédiction:", err);
      setError(err instanceof Error ? err.message : "Une erreur est survenue");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-6">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            🎾 TieBreaker IA - Prédiction
          </h1>
          <p className="text-gray-600">
            Prédisez l'issue d'un match de tennis grâce à l'intelligence artificielle
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Card */}
          <Card>
            <CardHeader>
              <CardTitle>Configuration du match</CardTitle>
              <CardDescription>
                Entrez les noms des deux joueurs pour obtenir une prédiction
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <User className="w-4 h-4" />
                  Joueur 1
                </label>
                <Input
                  placeholder="Ex: Rafael Nadal"
                  value={player1}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPlayer1(e.target.value)}
                  disabled={loading}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium flex items-center gap-2">
                  <User className="w-4 h-4" />
                  Joueur 2
                </label>
                <Input
                  placeholder="Ex: Roger Federer"
                  value={player2}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setPlayer2(e.target.value)}
                  disabled={loading}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  🎾 Surface du court
                </label>
                <select
                  value={surface}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setSurface(e.target.value)}
                  disabled={loading}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="Hard">Dur (Hard)</option>
                  <option value="Clay">Terre battue (Clay)</option>
                  <option value="Grass">Gazon (Grass)</option>
                  <option value="Carpet">Moquette (Carpet)</option>
                </select>
                <p className="text-xs text-gray-500">
                  La surface influence grandement les prédictions
                </p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  🏆 Tournoi
                </label>
                <Input
                  placeholder='Ex: Wimbledon'
                  value={tournament}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setTournament(e.target.value)}
                  disabled={loading}
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium">
                  🏅 Round
                </label>
                <select
                  value={round}
                  onChange={(e: React.ChangeEvent<HTMLSelectElement>) => setRound(e.target.value)}
                  disabled={loading}
                  className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <option value="">--</option>
                  <option value="F">Finale (F)</option>
                  <option value="SF">Demi-finale (SF)</option>
                  <option value="QF">Quart (QF)</option>
                  <option value="R16">Huitieme (R16)</option>
                  <option value="R32">Trente-deuxieme (R32)</option>
                  <option value="R64">Soixante-quatrieme (R64)</option>
                  <option value="R128">Cent-vingt-huitieme (R128)</option>
                </select>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    📅 Annee exacte
                  </label>
                  <Input
                    type="number"
                    min={1968}
                    max={2100}
                    placeholder="Ex: 2023"
                    value={year}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setYear(e.target.value)}
                    disabled={loading || allYears}
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-sm font-medium">
                    🗓️ Date exacte
                  </label>
                  <Input
                    type="date"
                    value={matchDate}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => setMatchDate(e.target.value)}
                    disabled={loading}
                  />
                </div>
              </div>

              <div className="flex items-center gap-2">
                <input
                  id="all-years"
                  type="checkbox"
                  className="h-4 w-4"
                  checked={allYears}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setAllYears(e.target.checked)}
                  disabled={loading}
                />
                <label htmlFor="all-years" className="text-sm font-medium">
                  Rechercher sur toutes les annees (plus lent)
                </label>
              </div>

              <Button
                onClick={handlePredict}
                disabled={loading || !player1 || !player2}
                className="w-full"
                size="lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Analyse en cours...
                  </>
                ) : (
                  <>
                    <TrendingUp className="mr-2 h-4 w-4" />
                    Lancer la prédiction
                  </>
                )}
              </Button>

              {error && (
                <div className="p-4 bg-red-50 border border-red-200 rounded-md text-red-800 text-sm">
                  {error}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Card */}
          <Card>
            <CardHeader>
              <CardTitle>Résultat de la prédiction</CardTitle>
              <CardDescription>
                Analyse basée sur les données historiques ATP
              </CardDescription>
            </CardHeader>
            <CardContent>
              {!result && !loading && (
                <div className="flex items-center justify-center h-64 text-gray-400">
                  <div className="text-center">
                    <TrendingUp className="w-12 h-12 mx-auto mb-2 opacity-20" />
                    <p>Aucune prédiction pour le moment</p>
                  </div>
                </div>
              )}

              {loading && (
                <div className="flex items-center justify-center h-64">
                  <Loader2 className="w-12 h-12 animate-spin text-primary" />
                </div>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="bg-gradient-to-r from-green-500 to-green-600 text-white rounded-lg p-6 text-center">
                    <p className="text-sm opacity-90 mb-2">Vainqueur prédit</p>
                    <p className="text-3xl font-bold">{result.winner_prediction}</p>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 mb-1">Joueur 1</p>
                      <p className="font-semibold">{result.player1}</p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-600 mb-1">Joueur 2</p>
                      <p className="font-semibold">{result.player2}</p>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-sm text-gray-600">Probabilité</span>
                      <span className="font-semibold">{(result.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${result.probability * 100}%` }}
                      />
                    </div>
                  </div>

                  <div className="bg-blue-50 border border-blue-200 rounded-md p-4">
                    <p className="text-sm text-blue-900">
                      <strong>Niveau de confiance:</strong> {result.confidence}
                    </p>
                  </div>

                  {/* AI Model Details */}
                  {result.details && result.details.model_info && (
                    <div className="bg-purple-50 border border-purple-200 rounded-md p-4">
                      <p className="text-sm font-semibold text-purple-900 mb-2">📊 Informations du Modèle IA</p>
                      <div className="grid grid-cols-2 gap-2 text-xs text-purple-800">
                        <div>
                          <span className="font-medium">Type:</span> {result.details.model_info.model_type || 'XGBoost'}
                        </div>
                        <div>
                          <span className="font-medium">Features:</span> {result.details.model_info.feature_count || 'N/A'}
                        </div>
                        <div>
                          <span className="font-medium">Année d'entraînement:</span> {result.details.model_info.train_end_year || 'N/A'}
                        </div>
                        <div>
                          <span className="font-medium">Surface:</span> {result.details.model_info.surface || 'N/A'}
                        </div>
                      </div>
                      <div className="mt-3 pt-3 border-t border-purple-200">
                        <p className="text-xs text-purple-700">
                          <strong>Probabilités détaillées:</strong>
                        </p>
                        <div className="grid grid-cols-2 gap-2 mt-2 text-xs">
                          <div className="bg-white bg-opacity-50 rounded px-2 py-1">
                            {result.player1}: {(result.details.p1_win_probability * 100).toFixed(1)}%
                          </div>
                          <div className="bg-white bg-opacity-50 rounded px-2 py-1">
                            {result.player2}: {(result.details.p2_win_probability * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Info Card */}
        <Card className="mt-6">
          <CardHeader>
            <CardTitle>Comment ça marche ?</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-bold">
                  1
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Données historiques</h4>
                  <p className="text-sm text-gray-600">
                    Analyse de milliers de matchs ATP depuis 1968
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-bold">
                  2
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Modèle XGBoost</h4>
                  <p className="text-sm text-gray-600">
                    Algorithme d'apprentissage automatique optimisé
                  </p>
                </div>
              </div>
              <div className="flex gap-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-full flex items-center justify-center text-green-600 font-bold">
                  3
                </div>
                <div>
                  <h4 className="font-semibold mb-1">Prédiction précise</h4>
                  <p className="text-sm text-gray-600">
                    Résultat basé sur les performances et statistiques
                  </p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

