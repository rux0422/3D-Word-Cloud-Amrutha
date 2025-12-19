/**
 * WordCloud3D Component
 * Main 3D visualization using React Three Fiber
 * Creates an immersive, interactive word cloud in 3D space
 */

import React, { useMemo, useRef, Suspense, useState, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, Text, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import type { WordData } from '../types';

// Tooltip data interface
interface TooltipData {
  word: string;
  weight: number;
  isNumber: boolean;
  x: number;
  y: number;
  sentence?: string;
  context?: string;
  entityType?: string;
}

// Configuration
const SPHERE_RADIUS = 9;
const MIN_FONT_SIZE = 0.25;
const MAX_FONT_SIZE = 1.4;
const ROTATION_SPEED = 0.0008;

interface WordCloud3DProps {
  words: WordData[];
}

interface Word3DItemProps {
  word: string;
  weight: number;
  position: [number, number, number];
  color: string;
  fontSize: number;
  isNumber: boolean;
  sentence?: string;
  context?: string;
  entityType?: string;
  onHover: (data: TooltipData | null) => void;
}

/**
 * Generates positions on a sphere using Fibonacci distribution
 * This creates an even distribution of points on a sphere
 */
function fibonacciSphere(
  index: number,
  total: number,
  radius: number
): [number, number, number] {
  const goldenRatio = (1 + Math.sqrt(5)) / 2;
  const theta = (2 * Math.PI * index) / goldenRatio;
  const phi = Math.acos(1 - (2 * (index + 0.5)) / total);

  // Add some randomness for a more organic feel
  const jitter = 0.15;
  const randRadius = radius * (1 + (Math.random() - 0.5) * jitter);

  const x = randRadius * Math.sin(phi) * Math.cos(theta);
  const y = randRadius * Math.sin(phi) * Math.sin(theta);
  const z = randRadius * Math.cos(phi);

  return [x, y, z];
}

/**
 * Generates vibrant color based on weight
 * Uses a beautiful rainbow gradient
 */
function getColor(weight: number, isNumber: boolean): string {
  if (isNumber) {
    // Numbers get special golden/amber colors
    const numberColors = [
      new THREE.Color('#FFD700'),  // Gold
      new THREE.Color('#FFA500'),  // Orange
      new THREE.Color('#FF8C00'),  // Dark Orange
      new THREE.Color('#FFB347'),  // Pastel Orange
    ];
    const idx = Math.floor(Math.random() * numberColors.length);
    return `#${numberColors[idx].getHexString()}`;
  }

  // Vibrant rainbow gradient for keywords
  const colors = [
    { weight: 0.00, color: new THREE.Color('#00D4FF') },  // Cyan
    { weight: 0.10, color: new THREE.Color('#00BFFF') },  // Deep Sky Blue
    { weight: 0.20, color: new THREE.Color('#1E90FF') },  // Dodger Blue
    { weight: 0.30, color: new THREE.Color('#6B5BFF') },  // Blue Violet
    { weight: 0.40, color: new THREE.Color('#9B59B6') },  // Purple
    { weight: 0.50, color: new THREE.Color('#E91E63') },  // Pink
    { weight: 0.60, color: new THREE.Color('#FF4757') },  // Red Pink
    { weight: 0.70, color: new THREE.Color('#FF6B35') },  // Orange Red
    { weight: 0.80, color: new THREE.Color('#FF9F43') },  // Orange
    { weight: 0.90, color: new THREE.Color('#FFD32D') },  // Yellow
    { weight: 1.00, color: new THREE.Color('#2ECC71') },  // Green (top weight)
  ];

  // Find interpolation range
  let lower = colors[0];
  let upper = colors[colors.length - 1];

  for (let i = 0; i < colors.length - 1; i++) {
    if (weight >= colors[i].weight && weight <= colors[i + 1].weight) {
      lower = colors[i];
      upper = colors[i + 1];
      break;
    }
  }

  // Interpolate between colors
  const t = (weight - lower.weight) / (upper.weight - lower.weight);
  const color = lower.color.clone().lerp(upper.color, t);

  return `#${color.getHexString()}`;
}

/**
 * Individual 3D Word with hover effects and animations
 */
const Word3DItem: React.FC<Word3DItemProps> = ({
  word,
  weight,
  position,
  color,
  fontSize,
  isNumber,
  sentence,
  context,
  entityType,
  onHover,
}) => {
  const textRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = React.useState(false);

  // Animation values
  const floatOffset = useMemo(() => Math.random() * Math.PI * 2, []);
  const floatSpeed = useMemo(() => 0.4 + Math.random() * 0.3, []);
  const floatAmplitude = useMemo(() => 0.08 + weight * 0.05, [weight]);

  useFrame((state) => {
    if (textRef.current) {
      // Gentle floating animation
      const time = state.clock.elapsedTime;
      const floatY = Math.sin(time * floatSpeed + floatOffset) * floatAmplitude;
      textRef.current.position.y = position[1] + floatY;

      // Rotate slightly for dynamic feel
      textRef.current.rotation.y = Math.sin(time * 0.3 + floatOffset) * 0.05;

      // Make text face camera (billboard effect)
      textRef.current.lookAt(state.camera.position);

      // Scale on hover with smooth lerp
      const targetScale = hovered ? 1.5 : 1;
      textRef.current.scale.lerp(
        new THREE.Vector3(targetScale, targetScale, targetScale),
        0.1
      );
    }
  });

  const glowIntensity = useMemo(() => 0.2 + weight * 0.6, [weight]);

  return (
    <Text
      ref={textRef}
      position={position}
      fontSize={fontSize}
      color={hovered ? '#ffffff' : color}
      anchorX="center"
      anchorY="middle"
      outlineWidth={hovered ? 0.04 : 0.015}
      outlineColor={color}
      onPointerOver={() => {
        setHovered(true);
        document.body.style.cursor = 'pointer';
        onHover({
          word,
          weight,
          isNumber,
          x: 0,
          y: 0,
          sentence,
          context,
          entityType,
        });
      }}
      onPointerOut={() => {
        setHovered(false);
        document.body.style.cursor = 'auto';
        onHover(null);
      }}
      onClick={(e) => {
        e.stopPropagation();
        console.log(`Word: "${word}" | Relevance: ${(weight * 100).toFixed(1)}% | Type: ${isNumber ? 'Number' : 'Keyword'}`);
      }}
    >
      {word}
      <meshStandardMaterial
        color={hovered ? '#ffffff' : color}
        emissive={color}
        emissiveIntensity={hovered ? glowIntensity * 2 : glowIntensity}
        metalness={0.15}
        roughness={0.35}
      />
    </Text>
  );
};

/**
 * Rotating group container for all words
 */
const WordCloudGroup: React.FC<{ words: WordData[]; onHover: (data: TooltipData | null) => void }> = ({ words, onHover }) => {
  const groupRef = useRef<THREE.Group>(null);

  // Prepare word elements with positions and styles
  const wordElements = useMemo(() => {
    return words.map((wordData, index) => {
      const position = fibonacciSphere(index, words.length, SPHERE_RADIUS);
      const fontSize = MIN_FONT_SIZE + wordData.weight * (MAX_FONT_SIZE - MIN_FONT_SIZE);
      const isNumber = wordData.type === 'number';
      const color = getColor(wordData.weight, isNumber);

      return {
        ...wordData,
        position: position as [number, number, number],
        fontSize,
        color,
        isNumber,
        key: `${wordData.word}-${index}`,
      };
    });
  }, [words]);

  // Slow rotation animation
  useFrame(() => {
    if (groupRef.current) {
      groupRef.current.rotation.y += ROTATION_SPEED;
    }
  });

  return (
    <group ref={groupRef}>
      {wordElements.map((wordEl) => (
        <Word3DItem
          key={wordEl.key}
          word={wordEl.word}
          weight={wordEl.weight}
          position={wordEl.position}
          color={wordEl.color}
          fontSize={wordEl.fontSize}
          isNumber={wordEl.isNumber}
          sentence={wordEl.sentence}
          context={wordEl.context}
          entityType={wordEl.entity_type}
          onHover={onHover}
        />
      ))}
    </group>
  );
};

/**
 * Ambient floating particles for atmosphere
 */
const Particles: React.FC = () => {
  const particlesRef = useRef<THREE.Points>(null);
  const count = 300;

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const colorPalette = [
      new THREE.Color('#FF6B6B'),
      new THREE.Color('#4ECDC4'),
      new THREE.Color('#45B7D1'),
      new THREE.Color('#96CEB4'),
      new THREE.Color('#FFEAA7'),
      new THREE.Color('#DDA0DD'),
    ];

    for (let i = 0; i < count; i++) {
      const radius = 18 + Math.random() * 12;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.random() * Math.PI;

      pos[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = radius * Math.cos(phi);

      const color = colorPalette[Math.floor(Math.random() * colorPalette.length)];
      col[i * 3] = color.r;
      col[i * 3 + 1] = color.g;
      col[i * 3 + 2] = color.b;
    }
    return [pos, col];
  }, []);

  useFrame((state) => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y = state.clock.elapsedTime * 0.015;
      particlesRef.current.rotation.x = Math.sin(state.clock.elapsedTime * 0.1) * 0.05;
    }
  });

  return (
    <points ref={particlesRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={count}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={count}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        size={0.08}
        vertexColors
        transparent
        opacity={0.7}
        sizeAttenuation
      />
    </points>
  );
};

/**
 * Scene lighting setup
 */
const Lighting: React.FC = () => {
  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[15, 15, 15]} intensity={1.2} color="#ffffff" />
      <pointLight position={[-15, -15, -15]} intensity={0.6} color="#FF6B6B" />
      <pointLight position={[0, 15, 0]} intensity={0.5} color="#4ECDC4" />
      <pointLight position={[15, -15, 15]} intensity={0.4} color="#FFD93D" />
      <pointLight position={[-15, 15, -15]} intensity={0.3} color="#9B59B6" />
    </>
  );
};

/**
 * Empty state placeholder
 */
const EmptyState: React.FC = () => {
  const groupRef = useRef<THREE.Group>(null);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.15;
      groupRef.current.position.y = Math.sin(state.clock.elapsedTime * 0.8) * 0.2;
    }
  });

  return (
    <group ref={groupRef}>
      <Text
        fontSize={1}
        color="#FF6B6B"
        anchorX="center"
        anchorY="middle"
        position={[0, 0.8, 0]}
      >
        3D Word Cloud
        <meshStandardMaterial
          color="#FF6B6B"
          emissive="#FF6B6B"
          emissiveIntensity={0.4}
        />
      </Text>
      <Text
        fontSize={0.4}
        color="#888888"
        anchorX="center"
        anchorY="middle"
        position={[0, -0.2, 0]}
      >
        Enter a URL above to visualize
      </Text>
      <Text
        fontSize={0.35}
        color="#666666"
        anchorX="center"
        anchorY="middle"
        position={[0, -0.7, 0]}
      >
        article topics in 3D space
      </Text>
    </group>
  );
};

/**
 * Get relevance level label based on weight
 */
function getRelevanceLevel(weight: number): { label: string; color: string } {
  if (weight >= 0.8) return { label: 'Very High', color: '#2ECC71' };
  if (weight >= 0.6) return { label: 'High', color: '#FF9F43' };
  if (weight >= 0.4) return { label: 'Medium', color: '#E91E63' };
  if (weight >= 0.2) return { label: 'Low', color: '#1E90FF' };
  return { label: 'Very Low', color: '#00D4FF' };
}

/**
 * Main WordCloud3D Component
 */
export const WordCloud3D: React.FC<WordCloud3DProps> = ({ words }) => {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null);
  const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
  const hoveredWordRef = useRef<TooltipData | null>(null);

  // Track mouse position globally and update tooltip position
  const handleMouseMove = useCallback((e: React.MouseEvent) => {
    const newPos = { x: e.clientX, y: e.clientY };
    setMousePos(newPos);
    // Update tooltip position if a word is hovered
    if (hoveredWordRef.current) {
      setTooltip({ ...hoveredWordRef.current, x: newPos.x, y: newPos.y });
    }
  }, []);

  const handleHover = useCallback((data: TooltipData | null) => {
    hoveredWordRef.current = data;
    if (data) {
      setTooltip({ ...data, x: mousePos.x, y: mousePos.y });
    } else {
      setTooltip(null);
    }
  }, [mousePos]);

  return (
    <div className="canvas-container" onMouseMove={handleMouseMove}>
      <Canvas>
        <PerspectiveCamera makeDefault position={[0, 0, 22]} fov={60} />

        <Suspense fallback={null}>
          <Lighting />
          <Stars
            radius={120}
            depth={60}
            count={3000}
            factor={5}
            saturation={0.3}
            fade
            speed={0.4}
          />
          <Particles />

          {words.length > 0 ? (
            <WordCloudGroup words={words} onHover={handleHover} />
          ) : (
            <EmptyState />
          )}
        </Suspense>

        <OrbitControls
          enablePan={false}
          enableZoom={true}
          minDistance={12}
          maxDistance={45}
          autoRotate={words.length === 0}
          autoRotateSpeed={0.4}
          dampingFactor={0.08}
          enableDamping
          rotateSpeed={0.5}
        />

        {/* Grid for depth perception */}
        <gridHelper
          args={[60, 60, '#1a1a3e', '#1a1a3e']}
          position={[0, -14, 0]}
        />
      </Canvas>

      {/* Tooltip for word relevance */}
      {tooltip && (
        <div
          className="word-tooltip"
          style={{
            left: Math.min(mousePos.x + 20, window.innerWidth - 380),
            top: Math.min(mousePos.y + 10, window.innerHeight - 300),
          }}
        >
          <div className="tooltip-word">{tooltip.word}</div>
          <div className="tooltip-relevance">
            <span className="tooltip-label">Relevance:</span>
            <span
              className="tooltip-value"
              style={{ color: getRelevanceLevel(tooltip.weight).color }}
            >
              {(tooltip.weight * 100).toFixed(1)}%
            </span>
          </div>
          <div className="tooltip-level">
            <span className="tooltip-label">Level:</span>
            <span
              className="tooltip-badge"
              style={{ backgroundColor: getRelevanceLevel(tooltip.weight).color }}
            >
              {getRelevanceLevel(tooltip.weight).label}
            </span>
          </div>
          {tooltip.context && (
            <div className="tooltip-context">
              <span className="tooltip-label">Meaning:</span>
              <span className="tooltip-context-text">{tooltip.context}</span>
            </div>
          )}
          {tooltip.sentence && (
            <div className="tooltip-sentence">
              <span className="tooltip-label">From article:</span>
              <span className="tooltip-sentence-text">"{tooltip.sentence}"</span>
            </div>
          )}
          <div className="tooltip-type">
            {tooltip.entityType ? `${tooltip.entityType}` : (tooltip.isNumber ? 'Statistic/Number' : 'Keyword')}
          </div>
        </div>
      )}
    </div>
  );
};

export default WordCloud3D;
