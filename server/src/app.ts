import express, { Express } from "express";
import bodyParser from "body-parser";
import cors from "cors";
import routes from "./routes/router";
import { Server as SocketIOServer } from "socket.io";
import Puppeteer, { Browser, Page, CDPSession } from "puppeteer";
import { installMouseHelper } from "./install-mouse-helper";
import fs from "fs";

const app = express();
const PORT = process.env.PORT || 4000;
const jsonParser = bodyParser.json();

app.use(cors());
app.use(routes);

const server = app.listen(PORT, () => {
    console.log("Server running on port:", PORT);
});

const io = new SocketIOServer(server);

class WebBrowserProxy {
    private d_browser?: Browser;
    private d_browserPage?: Page;
    private d_cdpSession?: CDPSession;
    constructor() {

    }
    public async start(socket: any) {
        if (this.d_browser) {
            await this.stop();
        }
        this.d_browser = await Puppeteer.launch({
            headless: true,
            args: [
                '--enable-usermedia-screen-capturing',
                '--allow-http-screen-capture',
                '--auto-select-desktop-capture-source=React App',

                '--use-fake-ui-for-media-stream',
                '--use-fake-device-for-media-stream',
                '--use-file-for-fake-audio-capture=/home/test.wav',
                '--allow-file-access'
            ],
            ignoreDefaultArgs: [
                '--mute-audio',
                '--hide-scrollbars'
            ]
         });
        const page = await this.d_browser.newPage();
        installMouseHelper(page);
        this.d_browserPage = page;
        page.on('console', message => console.log(message));
        const client = await page.target().createCDPSession();
        this.d_cdpSession = client;
        let cnt = 0;
        client.on('Page.screencastFrame', async (frame) => {
            console.log("Got frame:", cnt++);
            await client.send('Page.screencastFrameAck', { sessionId: frame.sessionId });
            //fs.writeFileSync('frame' + cnt + '.png', frame.data, 'base64');
            socket.emit('frame', frame.data)
        });
        await page.goto("http://google.com");
        await client.send('Page.startScreencast', {
            format: 'png', everyNthFrame: 1
        });
        console.log("Viewport:", page.viewport())
    }

    public async goto(url: string) {
        if (!this.d_browserPage) {
            throw new Error("No Browser Page");
        }
        console.log("Going to url:", url);
        await this.d_browserPage.goto(url);
        await this.d_browserPage.screenshot({ path: "test.png" });
    }

    public async mouseMove({ x, y } : { x:number, y:number }) {
        if (!this.d_browserPage) {
            throw new Error("No Browser Page");
        }
        console.log("Mouse move", { x, y});
        await this.d_browserPage.mouse.move(x, y);
    }

    public async mouseClick({ x, y } : { x: number, y:number }) {
        if (!this.d_browserPage) {
            throw new Error("No Browser Page");
        }
        console.log("Mouse click", { x, y});
        await this.d_browserPage.mouse.click(x, y);
    }

    public async scroll({ x=0, y=0 }) {
        await this.d_browserPage?.evaluate(({x,y}) => {
            window.scrollBy(x, y);
        }, {x,y});
    }

    public async stop() {
        console.log("WebBrowserProxy: Stopping");
        try {
            if (!this.d_cdpSession) {
                throw new Error("No CDP Session");
            }
            await this.d_cdpSession.send('Page.stopScreencast');
            
            await this.d_browserPage?.close();
            await this.d_browser?.close();
            this.d_cdpSession = undefined;
            this.d_browserPage = undefined;
            this.d_browser = undefined;
        }
        catch (error) {
            console.log("Failed to stop:", error);
        }
    }
};

const webBrowserProxy = new WebBrowserProxy();

// routes.post("/goto", jsonParser, async (req, resp) => {
//    console.log("REQ:", req.body.url);
//    await webBrowserProxy.goto(req.body.url);
//    resp.send();
// });

io.on("connection", (socket) => {
    console.log("Got socket.io connection");
    socket.on('startWebBrowser', async(cb) => {
        await webBrowserProxy.start(socket);
        cb();
    });
    socket.on('disconnecting', () => {
        console.log("Disconnecting");
        webBrowserProxy.stop();
    })
    socket.on('goto', async (url, cb) => {
        await webBrowserProxy.goto(url);
        cb();
    });
    socket.on('mouseMove', ({ x, y }) => {
        webBrowserProxy.mouseMove({x, y});
    });
    socket.on('mouseClick', ({ x, y }) => {
        webBrowserProxy.mouseClick({ x, y});
    });
    socket.on('scroll', ({ x, y }) => {
        webBrowserProxy.scroll({ x, y });
    });
});