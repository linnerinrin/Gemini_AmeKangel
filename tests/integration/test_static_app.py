def test_root_serves_boot_page(client):
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "boot" in response.text.lower()


def test_static_boot_page_is_mounted(client):
    response = client.get("/static/boot.html")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_static_javascript_is_mounted(client):
    response = client.get("/static/js/ui/desktop.js")

    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
