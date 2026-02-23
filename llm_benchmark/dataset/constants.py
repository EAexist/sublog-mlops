from pydantic import BaseModel

class CompanyInfo(BaseModel):
    name: str
    email: str
    industry: str

# Sample constants for dataset generation
COMPANIES = [
    CompanyInfo(name="Netflix", email="netflix@netflix.com", industry="OTT/Streaming"),
    CompanyInfo(name="Spotify", email="no-reply@spotify.com", industry="Music Streaming"),
]
