import type { ImageContent, Message, ModelImageResizeOptions, TextContent } from "@earendil-works/pi-ai";
import { formatDimensionNote, resizeImage, type ResizedImage } from "@earendil-works/pi-coding-agent";

const IMAGE_OMITTED_NOTE = "[Image omitted: could not be resized below the inline image size limit.]";

/**
 * Fits every image in an outgoing provider request within the model's image
 * resize profile, or pi's default (2000x2000 px, 4.5 MB of base64) when the
 * model has none. Anthropic rejects the whole request when it carries more than
 * 20 images and any of them exceeds 2000 px on a side, and every turn replays
 * the full history, so one oversized tool image (a Retina screenshot from Read,
 * an MCP capture) left a session failing on every later turn. Fitting at the
 * request covers every image source and repairs histories that already hold
 * such images; the persisted transcript keeps the originals. A resized image is
 * followed by pi's dimension note so the model can map coordinates back.
 *
 * Results are cached per image for the life of the fitter, so each image is
 * decoded once rather than on every request. Images are processed one at a
 * time: each resize runs in its own worker and holds the decoded bitmap.
 */
export class RequestImageFitter {
  private profile = "";
  private fitted = new Map<string, Promise<ResizedImage | null>>();

  async fit(messages: Message[], resize: ModelImageResizeOptions | undefined): Promise<Message[]> {
    const profile = JSON.stringify(resize ?? {});
    const previous = profile === this.profile ? this.fitted : new Map<string, Promise<ResizedImage | null>>();
    // Only images still in the request stay cached, so compacted ones are released.
    const fitted = new Map<string, Promise<ResizedImage | null>>();
    const out: Message[] = [];
    for (const message of messages) {
      if (
        (message.role !== "user" && message.role !== "toolResult") ||
        typeof message.content === "string" ||
        !message.content.some((block) => block.type === "image")
      ) {
        out.push(message);
        continue;
      }
      const content: Array<TextContent | ImageContent> = [];
      for (const block of message.content) {
        if (block.type !== "image") {
          content.push(block);
          continue;
        }
        const pending = fitted.get(block.data) ?? previous.get(block.data) ??
          resizeImage(Buffer.from(block.data, "base64"), block.mimeType, resize);
        fitted.set(block.data, pending);
        const resized = await pending;
        if (!resized) {
          content.push({ type: "text", text: IMAGE_OMITTED_NOTE });
        } else if (resized.data === block.data && resized.mimeType === block.mimeType) {
          content.push(block);
        } else {
          content.push({ type: "image", data: resized.data, mimeType: resized.mimeType });
          const note = formatDimensionNote(resized);
          if (note) content.push({ type: "text", text: note });
        }
      }
      out.push({ ...message, content } as Message);
    }
    this.profile = profile;
    this.fitted = fitted;
    return out;
  }
}
