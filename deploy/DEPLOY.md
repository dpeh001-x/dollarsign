# Deploying to predict.drdarylpeh.com

Five steps, ~15 minutes on a VPS that already runs nginx.

## 1. DNS record (at your domain registrar)

Add an **A record** pointing the subdomain at your VPS:

| Type | Name      | Value              |
|------|-----------|--------------------|
| A    | `predict` | `<your VPS IP>`    |

Wait until `ping predict.drdarylpeh.com` resolves to your server (usually minutes).

## 2. Get the code onto the server

```bash
git clone https://github.com/dpeh001-x/dollarsign.git
cd dollarsign
```

## 3. Build and start the app (Docker)

```bash
docker compose up -d --build
```

Verify it's alive:

```bash
curl http://127.0.0.1:8501/_stcore/health   # should print "ok"
```

The container binds to localhost only — nginx is the public door.

## 4. nginx reverse proxy

```bash
sudo cp deploy/nginx-predict.conf /etc/nginx/sites-available/predict
sudo ln -s /etc/nginx/sites-available/predict /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
```

At this point `http://predict.drdarylpeh.com` works.

## 5. HTTPS (Let's Encrypt)

```bash
sudo apt install -y certbot python3-certbot-nginx   # if not installed
sudo certbot --nginx -d predict.drdarylpeh.com
```

Certbot edits the nginx config for HTTPS and auto-renews. Done:
**https://predict.drdarylpeh.com**

## Updating to a new version

```bash
cd dollarsign
git pull
docker compose up -d --build
```

## Linking from DrDarylPeh.com

Add a link or button on the main site pointing to
`https://predict.drdarylpeh.com`. To embed it inside a page instead:

```html
<iframe src="https://predict.drdarylpeh.com/?embed=true"
        style="width:100%; height:900px; border:none;"></iframe>
```

## Troubleshooting

- **Blank page / stuck "connecting"** — the WebSocket headers in
  `nginx-predict.conf` are missing or nginx wasn't reloaded.
- **First prediction is slow** — the container fetches 4 years of price
  data on first request per symbol; it's cached (parquet volume) after that.
- **Container logs** — `docker logs -f dollarsign-predict`
