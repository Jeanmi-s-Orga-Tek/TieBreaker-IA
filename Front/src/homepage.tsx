import React from "react";

export default function HomePage() {
    return (
        <main className="container">
z            <header className="hero">
                <h1 className="hero-title">🎾 TieBreaker IA</h1>
                <p className="hero-subtitle">
                    Prédiction intelligente des résultats de matchs de tennis
                </p>
            </header>

            <section className="features">
                <div className="feature-card">
                    <div className="feature-icon">📊</div>
                    <h2>Analyse de Données</h2>
                    <p>Analyse approfondie de milliers de matchs ATP historiques</p>
                </div>

                <div className="feature-card">
                    <div className="feature-icon">🤖</div>
                    <h2>Intelligence Artificielle</h2>
                    <p>Modèles ML avancés (XGBoost) pour des prédictions précises</p>
                </div>

                <div className="feature-card">
                    <div className="feature-icon">⚡</div>
                    <h2>API Rapide</h2>
                    <p>Backend FastAPI performant avec réponses en temps réel</p>
                </div>
            </section>

            <section className="info-section">
                <h2 className="section-title">À Propos du Projet</h2>
                <div className="info-card">
                    <p>
                        TieBreaker-IA est un système de prédiction de résultats de matchs de tennis
                        basé sur l'apprentissage automatique. Il utilise des données historiques ATP
                        pour entraîner des modèles prédictifs sophistiqués.
                    </p>
                    <ul className="feature-list">
                        <li>✅ Données ATP complètes depuis 1968</li>
                        <li>✅ Multiples modèles ML (Random Forest, XGBoost)</li>
                        <li>✅ Interface web moderne avec React</li>
                        <li>✅ API RESTful avec FastAPI</li>
                        <li>✅ Déploiement Docker optimisé</li>
                    </ul>
                </div>
            </section>

            <section className="api-section">
                <h2 className="section-title">API Endpoints</h2>
                <div className="endpoints">
                    <a href="/api/health" className="endpoint-link">
                        <span className="endpoint-method">GET</span>
                        <span className="endpoint-path">/api/health</span>
                        <span className="endpoint-desc">Vérifier l'état de l'API</span>
                    </a>
                </div>
            </section>

            <footer className="footer">
                <p>© 2026 TieBreaker-IA | Epitech Tek3 Project</p>
            </footer>
        </main>
    );
}