FROM node:22-slim

ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
ENV CI=true

RUN corepack enable && corepack prepare pnpm@11.4.0 --activate

# Real Google Chrome + xvfb (virtual display for headed mode) + Chrome libs
RUN apt-get update && apt-get install -y --no-install-recommends \
      xvfb xauth wget ca-certificates fonts-liberation \
      libasound2 libatk-bridge2.0-0 libatk1.0-0 libcups2 libdbus-1-3 \
      libdrm2 libgbm1 libnspr4 libnss3 libxcomposite1 libxdamage1 \
      libxfixes3 libxkbcommon0 libxrandr2 libxss1 xdg-utils \
    && wget -q -O /tmp/chrome.deb \
       https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/chrome.deb \
    && rm /tmp/chrome.deb && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app
COPY package.json tsconfig.json ./
RUN pnpm install

COPY . .
RUN pnpm run build

RUN chown -R node:node /usr/src/app

ARG PORT=3000
ENV PORT=${PORT}
EXPOSE ${PORT}

USER node

# Headed Chrome needs a display — start Xvfb ourselves; xvfb-run hangs in containers
CMD ["sh", "-c", "Xvfb :99 -screen 0 1920x1080x24 & sleep 1; DISPLAY=:99 exec node dist/api.js"]
