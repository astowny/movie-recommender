import { useState } from 'react'

const API_BASE = 'http://localhost:8000'

function App() {
  const [userId, setUserId] = useState('1')
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchRecommendations = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE}/recommend/${userId}`)
      const json = await response.json()
      if (response.ok) {
        setRecommendations(json.recommendations || [])
      } else {
        setError(json.error || 'Erreur inconnue')
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header>
        <h1>Movie Recommender</h1>
        <p>Testez des recommandations de films pour un utilisateur.</p>
      </header>
      <main>
        <div className="panel">
          <label>
            User ID:
            <input
              type="number"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              min="1"
            />
          </label>
          <button onClick={fetchRecommendations} disabled={loading}>
            {loading ? 'Chargement...' : 'Obtenir des recommandations'}
          </button>
          {error && <div className="error">{error}</div>}
        </div>
        <section>
          <h2>Recommandations</h2>
          {recommendations.length === 0 ? (
            <p>Aucune recommandation pour l'instant.</p>
          ) : (
            <ul>
              {recommendations.map((item) => (
                <li key={item.movieId}>
                  <strong>{item.title || `Film #${item.movieId}`}</strong>
                  <span>score {item.score.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          )}
        </section>
      </main>
    </div>
  )
}

export default App
