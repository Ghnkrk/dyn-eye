"use client";

import type { UpsamplingBox } from "@/lib/types";

// Distinct, readable on dark and light steel images
const BOX_COLORS = ["#f59e0b", "#22d3ee", "#a3e635", "#f472b6", "#818cf8", "#fb7185", "#34d399", "#facc15", "#60a5fa", "#e879f9"];

export function boxColor(classId: number) {
  return BOX_COLORS[classId % BOX_COLORS.length];
}

interface BoxedImageProps {
  src: string;
  boxes: UpsamplingBox[];
  alt?: string;
  className?: string;
  showLabels?: boolean;
}

/** An image with its YOLO boxes drawn on top (the overlay scales with the image). */
export function BoxedImage({ src, boxes, alt = "", className, showLabels = false }: BoxedImageProps) {
  return (
    <div className={`relative ${className ?? ""}`}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} className="block h-auto w-full rounded" loading="lazy" />
      <svg className="pointer-events-none absolute inset-0 h-full w-full" viewBox="0 0 1 1" preserveAspectRatio="none">
        {boxes.map((b, i) => {
          const [x, y, w, h] = b.xywhn;
          return (
            <rect
              key={i}
              x={x - w / 2}
              y={y - h / 2}
              width={w}
              height={h}
              fill="none"
              stroke={boxColor(b.class_id)}
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          );
        })}
      </svg>
      {showLabels &&
        boxes.map((b, i) => {
          const [x, y, w, h] = b.xywhn;
          return (
            <span
              key={i}
              className="pointer-events-none absolute -translate-y-full rounded-sm px-1 text-[10px] font-medium leading-4 text-black"
              style={{
                left: `${Math.max(0, (x - w / 2) * 100)}%`,
                top: `${Math.max(0, (y - h / 2) * 100)}%`,
                backgroundColor: boxColor(b.class_id),
              }}
            >
              {b.class_name}
            </span>
          );
        })}
    </div>
  );
}
