<?php

namespace Pankaj\MHFSafeguard\Content;

class ContentContext
{
    protected $contentType = 'post';
    protected $contentId = 0;
    protected $threadId = 0;
    protected $nodeId = 0;
    protected $userId = 0;
    protected $username = '';
    protected $title = '';
    protected $message = '';
    protected $isFirstPost = false;

    public function __construct(array $values = [])
    {
        foreach ($values as $key => $value)
        {
            if (property_exists($this, $key))
            {
                $this->{$key} = $value;
            }
        }
    }

    public function toPayloadArray(): array
    {
        return [
            'content_type' => $this->contentType,
            'content_id' => (int)$this->contentId,
            'thread_id' => (int)$this->threadId,
            'node_id' => (int)$this->nodeId,
            'user_id' => (int)$this->userId,
            'username' => (string)$this->username,
            'title' => (string)$this->title,
            'is_first_post' => (bool)$this->isFirstPost,
        ];
    }

    public function getContentType(): string { return $this->contentType; }
    public function getContentId(): int { return (int)$this->contentId; }
    public function getThreadId(): int { return (int)$this->threadId; }
    public function getNodeId(): int { return (int)$this->nodeId; }
    public function getUserId(): int { return (int)$this->userId; }
    public function getUsername(): string { return (string)$this->username; }
    public function getTitle(): string { return (string)$this->title; }
    public function getMessage(): string { return (string)$this->message; }
    public function isFirstPost(): bool { return (bool)$this->isFirstPost; }

    public function withContentId(int $contentId): self
    {
        $copy = clone $this;
        $copy->contentId = $contentId;
        return $copy;
    }

    public function withThreadId(int $threadId): self
    {
        $copy = clone $this;
        $copy->threadId = $threadId;
        return $copy;
    }
}
