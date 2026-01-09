import React from "react";

export default function HomePage() {
    return (
        <main className="container">
            <h1 className="title">Home</h1>

            <section className="card">
                <p>
                    Backend route: <code>/home</code>
                </p>
                <p>
                    Health: <a href="/api/health">/api/health</a>
                </p>
            </section>
        </main>
    );
}