def test_root_returns_boot_html(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "NEEDY GIRL OVERDOSE" in response.text
    assert "js/boot.js" in response.text
