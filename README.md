# Airline Passenger Satisfaction

Deployment branch for the existing Streamlit website. App source, data, trained
pipeline and verified Tableau results are unchanged copies of the local project.

## Streamlit Community Cloud settings

- Repository: `Salama1122/Airline_Passenger_Satisfaction`
- Branch: `deploy/streamlit`
- Main file path: `app.py`
- Advanced settings → Python: `3.13`
- Secrets: none required

The website uses saved artifacts; it does not train models during deployment.
Only the hosting bind address and dependency selection differ from the local setup.
`deployment_manifest.json` records file checksums. Update this branch to update the hosted app.
